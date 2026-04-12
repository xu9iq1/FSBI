import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import Detector
from utils.esbi import ESBI_Dataset
from utils.funcs import load_json
from utils.runtime import get_device, seed_everything


def compute_accuracy(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().numpy()
    return float((pred_idx == true.cpu().numpy()).mean())


def main(args):
    cfg = load_json(args.config)
    device = get_device(args.device)
    seed_everything(args.seed, device)

    dataset = ESBI_Dataset(
        phase=args.phase,
        image_size=cfg["image_size"],
        wavelet=args.wavelet,
        mode=args.mode,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=dataset.worker_init_fn,
    )

    model = Detector().to(device)
    checkpoint = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    outputs = []
    targets = []

    for data in tqdm(loader):
        img = data["img"].to(device, non_blocking=device.type == "cuda").float()
        target = data["label"].to(device, non_blocking=device.type == "cuda").long()

        with torch.no_grad():
            pred = model(img)
            loss = criterion(pred, target)

        total_loss += loss.item()
        total_acc += compute_accuracy(F.log_softmax(pred, dim=1), target)
        outputs += pred.softmax(1)[:, 1].cpu().numpy().tolist()
        targets += target.cpu().numpy().tolist()

    auc = roc_auc_score(targets, outputs)
    print(
        f"{args.phase} | device={device.type} | "
        f"loss={total_loss / len(loader):.4f} | "
        f"acc={total_acc / len(loader):.4f} | auc={auc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("-w", "--weight", required=True)
    parser.add_argument("--phase", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("-m", "--mode", default="reflect")
    parser.add_argument("-t", "--wavelet", default="sym2")
    main(parser.parse_args())
