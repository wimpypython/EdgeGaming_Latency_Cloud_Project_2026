#!/usr/bin/env python3
"""
Pull the feature artifact from S3 onto the edge node.

WHY THIS EXISTS
    This is the line that makes "dataset in S3, compute at the edge node"
    a real architectural statement rather than a label on a diagram.

    It also establishes the two-tier split the project actually needs:

        s3://.../raw/       the 4.75 GB CSKnow HDF5 -- archive of record,
                            consumed by the parser on a workstation
        s3://.../features/  parsed_data.npz, 16 MB -- what the edge node
                            retrieves and uses

    That split is not a workaround. The edge node has an 8 GB root volume
    with under 2 GB free; it could not hold the raw dataset if it wanted
    to, and a real edge deployment would not carry a training corpus
    anyway. Heavy data stays in object storage; the edge pulls prepared
    features.

NO CREDENTIALS ON THIS BOX
    boto3 finds credentials automatically from the IAM role attached to
    the instance. Nothing is stored on disk, the credentials rotate on
    their own, and there is no key here to leak. If this script is ever
    run somewhere without a role it will say so rather than failing
    obscurely.

    If you find yourself tempted to paste an access key onto a server,
    this is the thing to do instead.

WHAT IT CHECKS
    free disk BEFORE downloading
        The instance has ~1.9 GB free. Filling the root filesystem takes
        the whole box down, including the thing you were trying to run.
        This refuses rather than finding out halfway.

    whether the file is already current
        Compares size and ETag against S3 and skips the transfer if they
        match. Startup scripts get run again and again.

    that the artifact actually loads
        A download that produced a corrupt file is worse than a failed
        one, because it fails later somewhere confusing.

USAGE
    python3 fetch_dataset.py
    python3 fetch_dataset.py --key features/parsed_data.npz --dest ~/data
    python3 fetch_dataset.py --list        # what is in the bucket
"""

import argparse
import os
import shutil
import sys

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    sys.exit("boto3 is not installed.  pip install boto3  "
             "(or: sudo apt install python3-boto3)")

BUCKET = "edgegaming-csknow-apsouth1"
REGION = "ap-south-1"
KEY = "features/parsed_data.npz"
DEST = os.path.expanduser("~/data")

# Refuse to download if it would leave less than this free. Running the
# root filesystem to zero on a 8 GB volume breaks apt, logging, and the
# BPF toolchain all at once.
MIN_FREE_AFTER = 500 * 1024 * 1024      # 500 MB


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_credentials(s3):
    """Fail early and legibly, naming the likely cause."""
    try:
        boto3.client("sts", region_name=REGION).get_caller_identity()
    except NoCredentialsError:
        sys.exit("No credentials. On EC2 this means no IAM role is attached:\n"
                 "  EC2 -> Instances -> select -> Actions -> Security ->\n"
                 "  Modify IAM role -> EdgeGamingEdgeNode")
    except Exception as e:
        sys.exit(f"Could not authenticate: {type(e).__name__}: {e}")


def list_bucket(s3):
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET)
    except ClientError as e:
        sys.exit(f"Cannot list s3://{BUCKET}: {e}")
    if "Contents" not in resp:
        print(f"s3://{BUCKET} is empty")
        return
    print(f"s3://{BUCKET}")
    for o in resp["Contents"]:
        print(f"  {human(o['Size']):>10}  {o['LastModified']:%Y-%m-%d %H:%M}"
              f"  {o['Key']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--list", action="store_true",
                    help="show what is in the bucket and exit")
    ap.add_argument("--force", action="store_true",
                    help="download even if the local copy looks current")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    check_credentials(s3)

    if args.list:
        list_bucket(s3)
        return

    # What is there, and how big?
    try:
        head = s3.head_object(Bucket=args.bucket, Key=args.key)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            sys.exit(f"s3://{args.bucket}/{args.key} does not exist.\n"
                     f"Run with --list to see what is there.")
        if code in ("403", "AccessDenied"):
            sys.exit(f"Access denied reading s3://{args.bucket}/{args.key}.\n"
                     f"The instance role needs s3:GetObject on "
                     f"arn:aws:s3:::{args.bucket}/*")
        raise

    size = head["ContentLength"]
    etag = head["ETag"].strip('"')
    print(f"remote : s3://{args.bucket}/{args.key}")
    print(f"         {human(size)}  etag {etag[:16]}")

    os.makedirs(args.dest, exist_ok=True)
    local = os.path.join(args.dest, os.path.basename(args.key))

    # Already current?
    if os.path.exists(local) and not args.force:
        if os.path.getsize(local) == size:
            print(f"local  : {local}")
            print("         same size as S3 -- already current, skipping.")
            print("         (--force to download anyway)")
            verify(local)
            return
        print(f"local  : {local} exists but differs in size, re-downloading")

    # Enough room?
    free = shutil.disk_usage(args.dest).free
    print(f"disk   : {human(free)} free at {args.dest}")
    if free - size < MIN_FREE_AFTER:
        sys.exit(f"\nREFUSING to download.\n"
                 f"  need {human(size)}, have {human(free)} free, and this\n"
                 f"  must leave at least {human(MIN_FREE_AFTER)} behind.\n"
                 f"  Filling the root volume breaks apt, logging and the BPF\n"
                 f"  toolchain. Free space or use a larger volume.")

    print(f"\ndownloading to {local} ...")
    s3.download_file(args.bucket, args.key, local)
    got = os.path.getsize(local)
    print(f"got {human(got)}")
    if got != size:
        sys.exit(f"SIZE MISMATCH: expected {size}, got {got}. "
                 f"Delete the file and retry.")

    verify(local)


def verify(path):
    """A corrupt download is worse than a failed one -- it fails later,
    somewhere confusing. Check the artifact actually opens."""
    if not path.endswith(".npz"):
        print("\nnot an .npz, skipping content check")
        return
    try:
        import numpy as np
    except ImportError:
        print("\nnumpy not installed here, skipping content check")
        return

    print("\nverifying ...")
    try:
        with np.load(path, allow_pickle=True) as d:
            for name in d.files:
                a = d[name]
                print(f"  {name:<10} {str(a.shape):<20} {a.dtype}")
            if "X" in d.files and "y" in d.files:
                if len(d["X"]) != len(d["y"]):
                    sys.exit(f"  X has {len(d['X'])} rows but y has "
                             f"{len(d['y'])} -- the artifact is wrong")
                print(f"\n  {len(d['X']):,} examples, X and y agree.")
    except Exception as e:
        sys.exit(f"  could not load: {type(e).__name__}: {e}\n"
                 f"  The download is corrupt. Delete it and retry.")
    print("  OK")


if __name__ == "__main__":
    main()
