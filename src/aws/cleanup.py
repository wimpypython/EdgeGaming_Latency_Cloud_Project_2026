#!/usr/bin/env python3
"""
Tear down the project's AWS resources, reproducibly.

WHY THIS EXISTS
    Two reasons, and the second is the one that matters for the report.

    1. Nothing should bill after the project ends.
    2. "We provisioned it and we can deprovision it with one command" is
       an infrastructure claim. A system that can only be built by
       clicking through a console once is a configuration someone did,
       not an architecture.

SAFETY -- read this before running
    This script DELETES THINGS AND CANNOT UNDO THEM.

    By default it deletes nothing. It finds the resources, prints what it
    would remove, and exits. You must pass --yes to make it act, and even
    then it prints the list and pauses.

    It also refuses to touch anything that is not tagged as belonging to
    this project, so a mistyped filter deletes nothing rather than
    something else.

WHY TAGS AND NOT IDS
    Hardcoding i-07de4e55202b7dba7 works until the instance is recreated,
    and then the script either fails or -- much worse -- the ID has been
    reused and it deletes a stranger's resource. Tag-based discovery is
    correct at any point in the project's life.

    If your resources are not tagged yet, tag them once:

        aws ec2 create-tags --resources <instance-id> <sg-id> \\
            --tags Key=Project,Value=EdgeGaming

ORDER OF OPERATIONS -- this is the part that bites
    A security group cannot be deleted while an instance still uses it,
    and terminating an instance is not instant: it goes
    running -> shutting-down -> terminated over tens of seconds. So the
    order is

        terminate instances
        WAIT until they are actually terminated
        delete security groups
        delete key pairs

    Skipping the wait leaves the security group behind with a
    DependencyViolation, and an orphaned security group is the kind of
    thing nobody notices for months.

WHAT IT LEAVES ALONE BY DEFAULT
    The $5 budget alarm. You generally want billing alerts to outlive the
    project. Pass --budget to remove it too.

    The S3 bucket, unless you pass --bucket NAME. Buckets must be emptied
    before they can be deleted, including versioned objects, so that is
    handled explicitly rather than hopefully.

REQUIREMENTS
    pip install boto3
    AWS credentials configured (aws configure, or env vars, or an IAM
    role). The script checks and says so rather than failing obscurely.

USAGE
    python3 cleanup.py                      # dry run, shows what it finds
    python3 cleanup.py --yes                # actually tear down
    python3 cleanup.py --yes --bucket edgegaming-dataset-apsouth1
    python3 cleanup.py --region ap-south-1 --tag Project=EdgeGaming
"""

import argparse
import sys
import time

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    sys.exit("boto3 is not installed.  pip install boto3")

DEFAULT_REGION = "ap-south-1"
DEFAULT_TAG = "Project=EdgeGaming"

# How long to wait for instances to reach 'terminated' before giving up
# and refusing to delete the security group. Termination normally takes
# 30-90 s; 5 minutes is generous without hanging forever.
TERMINATE_TIMEOUT_S = 300


def banner(text):
    print()
    print(text)
    print("-" * len(text))


def parse_tag(s):
    if "=" not in s:
        sys.exit(f"--tag must be Key=Value, got {s!r}")
    k, v = s.split("=", 1)
    return k, v


def find_instances(ec2, key, value):
    """Instances with the tag, excluding ones already terminated."""
    resp = ec2.describe_instances(Filters=[
        {"Name": f"tag:{key}", "Values": [value]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    out = []
    for r in resp["Reservations"]:
        for i in r["Instances"]:
            name = next((t["Value"] for t in i.get("Tags", [])
                         if t["Key"] == "Name"), "")
            out.append({"id": i["InstanceId"], "name": name,
                        "type": i["InstanceType"],
                        "state": i["State"]["Name"]})
    return out


def find_security_groups(ec2, key, value):
    resp = ec2.describe_security_groups(Filters=[
        {"Name": f"tag:{key}", "Values": [value]},
    ])
    # 'default' cannot be deleted and must never be attempted.
    return [{"id": g["GroupId"], "name": g["GroupName"]}
            for g in resp["SecurityGroups"] if g["GroupName"] != "default"]


def find_key_pairs(ec2, key, value):
    try:
        resp = ec2.describe_key_pairs(Filters=[
            {"Name": f"tag:{key}", "Values": [value]},
        ])
    except ClientError:
        return []
    return [{"name": k["KeyName"]} for k in resp["KeyPairs"]]


def find_volumes(ec2, key, value):
    """Volumes not attached to anything. A detached volume bills quietly
    forever and is the most commonly forgotten resource in a teardown."""
    resp = ec2.describe_volumes(Filters=[
        {"Name": f"tag:{key}", "Values": [value]},
        {"Name": "status", "Values": ["available"]},
    ])
    return [{"id": v["VolumeId"], "size": v["Size"]} for v in resp["Volumes"]]


def wait_terminated(ec2, ids):
    print(f"  waiting for {len(ids)} instance(s) to terminate "
          f"(up to {TERMINATE_TIMEOUT_S}s)...")
    deadline = time.time() + TERMINATE_TIMEOUT_S
    while time.time() < deadline:
        resp = ec2.describe_instances(InstanceIds=ids)
        states = [i["State"]["Name"]
                  for r in resp["Reservations"] for i in r["Instances"]]
        if all(s == "terminated" for s in states):
            print("  all terminated")
            return True
        print(f"    {', '.join(sorted(set(states)))} ...")
        time.sleep(10)
    print("  TIMED OUT waiting for termination.")
    print("  Not deleting security groups -- they would fail with")
    print("  DependencyViolation and leave things half-removed.")
    return False


def empty_and_delete_bucket(s3, bucket):
    """Buckets must be empty before deletion, including old versions and
    delete markers if versioning was ever enabled."""
    print(f"  emptying s3://{bucket}")
    paginator = s3.get_paginator("list_object_versions")
    n = 0
    try:
        for page in paginator.paginate(Bucket=bucket):
            objs = [{"Key": o["Key"], "VersionId": o["VersionId"]}
                    for o in page.get("Versions", [])
                    + page.get("DeleteMarkers", [])]
            if objs:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
                n += len(objs)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            print(f"  bucket {bucket} does not exist, skipping")
            return
        raise
    print(f"  deleted {n} object version(s)")
    s3.delete_bucket(Bucket=bucket)
    print(f"  deleted bucket {bucket}")


def delete_budget(region, name):
    budgets = boto3.client("budgets", region_name="us-east-1")
    acct = boto3.client("sts").get_caller_identity()["Account"]
    budgets.delete_budget(AccountId=acct, BudgetName=name)
    print(f"  deleted budget {name}")


def main():
    ap = argparse.ArgumentParser(
        description="Tear down EdgeGaming AWS resources. Dry run unless --yes.")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"Key=Value selecting this project's resources "
                         f"(default {DEFAULT_TAG})")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete. Without this, nothing is removed.")
    ap.add_argument("--bucket", help="also empty and delete this S3 bucket")
    ap.add_argument("--budget", help="also delete this budget by name "
                                     "(you usually want to keep it)")
    args = ap.parse_args()

    key, value = parse_tag(args.tag)

    # One cheap call to fail fast and legibly. Everything that can go
    # wrong here -- no credentials, expired credentials, no network, a
    # bad region -- otherwise surfaces much later as a stack trace in the
    # middle of a teardown, which is the worst possible moment.
    try:
        ec2 = boto3.client("ec2", region_name=args.region)
        ec2.describe_regions(RegionNames=[args.region])
    except NoCredentialsError:
        sys.exit("No AWS credentials found. Run 'aws configure' first.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AuthFailure", "UnauthorizedOperation",
                    "RequestExpired", "InvalidClientTokenId"):
            sys.exit(f"AWS rejected the credentials ({code}). "
                     f"Run 'aws configure' or refresh them.")
        sys.exit(f"AWS rejected the request: {e}")
    except Exception as e:
        sys.exit(f"Could not reach AWS in {args.region}: "
                 f"{type(e).__name__}: {e}")

    print(f"region  : {args.region}")
    print(f"selector: {key}={value}")
    print(f"mode    : {'DELETE' if args.yes else 'dry run (nothing will be removed)'}")

    instances = find_instances(ec2, key, value)
    groups = find_security_groups(ec2, key, value)
    keys = find_key_pairs(ec2, key, value)
    volumes = find_volumes(ec2, key, value)

    banner("Found")
    for i in instances:
        print(f"  instance        {i['id']}  {i['type']:<12} "
              f"{i['state']:<10} {i['name']}")
    for g in groups:
        print(f"  security group  {g['id']}  {g['name']}")
    for k in keys:
        print(f"  key pair        {k['name']}")
    for v in volumes:
        print(f"  volume          {v['id']}  {v['size']} GiB  (unattached)")
    if args.bucket:
        print(f"  s3 bucket       {args.bucket}")
    if args.budget:
        print(f"  budget          {args.budget}")

    total = len(instances) + len(groups) + len(keys) + len(volumes)
    if total == 0 and not args.bucket and not args.budget:
        print("  nothing tagged "
              f"{key}={value} in {args.region}.")
        print("\nIf resources exist but are untagged, tag them first:")
        print(f"  aws ec2 create-tags --resources <id> ... "
              f"--tags Key={key},Value={value}")
        return

    if not args.yes:
        print("\nDry run. Nothing was deleted. Re-run with --yes to proceed.")
        return

    print("\nThis cannot be undone. Type the region name to confirm: ", end="")
    if input().strip() != args.region:
        sys.exit("Did not match. Nothing deleted.")

    banner("Deleting")

    ids = [i["id"] for i in instances]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
        print(f"  terminate requested for {', '.join(ids)}")
        if not wait_terminated(ec2, ids):
            sys.exit(1)

    for g in groups:
        try:
            ec2.delete_security_group(GroupId=g["id"])
            print(f"  deleted security group {g['id']}")
        except ClientError as e:
            print(f"  could not delete {g['id']}: {e}")

    for k in keys:
        ec2.delete_key_pair(KeyName=k["name"])
        print(f"  deleted key pair {k['name']}")
        print(f"    (the local .pem file is yours to remove)")

    for v in volumes:
        ec2.delete_volume(VolumeId=v["id"])
        print(f"  deleted volume {v['id']}")

    if args.bucket:
        empty_and_delete_bucket(boto3.client("s3", region_name=args.region),
                                args.bucket)

    if args.budget:
        delete_budget(args.region, args.budget)

    banner("Done")
    print("  Check the console once: tagging only finds what is tagged.")


if __name__ == "__main__":
    main()
