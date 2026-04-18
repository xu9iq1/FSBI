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
    train_acc = []
    val_acc = []

    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.match(line.strip())
        if not match:
            continue

        epoch, _, _, tr_acc, _, va_acc, _ = match.groups()
        epochs.append(int(epoch))
        train_acc.append(float(tr_acc))
        val_acc.append(float(va_acc))

    if not epochs:
        raise ValueError(f"No epoch metrics found in {log_path}")

    return {
        "epochs": epochs,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }


def configure_font(font_path=None):
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


def plot_curves(metrics, output_path, font_path=None):
    configure_font(font_path)
    epochs = metrics["epochs"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=200)
    ax.plot(
        epochs,
        metrics["train_acc"],
        color="#c84b31",
        linewidth=2,
        linestyle="-",
        label="训练准确率",
    )
    ax.plot(
        epochs,
        metrics["val_acc"],
        color="#1f6f8b",
        linewidth=2,
        linestyle="--",
        label="验证准确率",
    )
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("准确率")
    ax.set_title("训练准确率与验证准确率变化曲线")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
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
        default="output/full_base_04_13_17_35_59/figures/fig4_1_accuracy_curves.png",
    )
    parser.add_argument("--font-path")
    args = parser.parse_args()

    metrics = load_metrics(args.log_path)
    plot_curves(metrics, args.output_path, font_path=args.font_path)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
