import torch, glob
for f in sorted(glob.glob("gru_h*.pt")):
    c = torch.load(f, map_location="cpu", weights_only=False)
    gain = (1 - c["val_median"] / c["dr_median"]) * 100
    print("%-18s hidden=%4d layers=%d h=%3d val=%7.3f dr=%7.3f gain=%5.1f%%"
          % (f, c["hidden"], c["layers"], c["horizon"],
             c["val_median"], c["dr_median"], gain))
