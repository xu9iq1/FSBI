import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


LOG_PATTERN = re.compile(
    r"Epoch (\d+)/(\d+) \| "
    r"train loss: ([0-9.]+), train acc: ([0-9.]+), "
    r"val loss: ([0-9.]+), val acc: ([0-9.]+), val auc: ([0-9.]+)"
)


def load_metrics(log_path):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_auc = []

    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.match(line.strip())
        if not match:
            continue

        epoch, _, tr_loss, tr_acc, va_loss, va_acc, va_auc = match.groups()
        epochs.append(int(epoch))
        train_loss.append(float(tr_loss))
        train_acc.append(float(tr_acc))
        val_loss.append(float(va_loss))
        val_acc.append(float(va_acc))
        val_auc.append(float(va_auc))

    if not epochs:
        raise ValueError(f"No epoch metrics found in {log_path}")

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auc": val_auc,
    }


def plot_curves(metrics, output_path, font_path=None):
    epochs = metrics["epochs"]
    val_auc = metrics["val_auc"]

    best_idx = max(range(len(val_auc)), key=val_auc.__getitem__)
    best_epoch = epochs[best_idx]
    best_auc = val_auc[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)

    if font_path:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "SimHei",
            "Microsoft YaHei",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False

    axes[0].plot(epochs, metrics["train_loss"], color="#d1495b", linewidth=2)
    axes[0].set_title("(a) 训练损失曲线")
    axes[0].set_xlabel("训练轮数")
    axes[0].set_ylabel("训练损失")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, val_auc, color="#00798c", linewidth=2)
    axes[1].scatter([best_epoch], [best_auc], color="#edae49", s=40, zorder=3)
    axes[1].set_title("(b) 验证 AUC 曲线")
    axes[1].set_xlabel("训练轮数")
    axes[1].set_ylabel("验证 AUC")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        default="output/full_base_04_13_17_35_59/logs/losses.logs",
    )
    parser.add_argument(
        "--output-path",
        default="output/full_base_04_13_17_35_59/figures/fig4_2_training_curves.png",
    )
    parser.add_argument("--font-path")
    args = parser.parse_args()

    metrics = load_metrics(args.log_path)
    globals()["args"] = args
    plot_curves(metrics, args.output_path, font_path=args.font_path)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
