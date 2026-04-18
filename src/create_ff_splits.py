import argparse
import json
import random
from pathlib import Path


def pair_ids(video_ids):
    return [video_ids[i : i + 2] for i in range(0, len(video_ids), 2)]


def split_ids(ids, train_videos, val_videos, test_videos):
    train_ids = ids[:train_videos]
    val_ids = ids[train_videos : train_videos + val_videos]
    test_ids = ids[train_videos + val_videos : train_videos + val_videos + test_videos]
    return train_ids, val_ids, test_ids


def main(args):
    video_dir = Path(args.video_dir)
    ids = sorted(path.stem for path in video_dir.glob("*.mp4"))

    total_required = args.train_videos + args.val_videos + args.test_videos
    if total_required > len(ids):
        raise ValueError(
            f"Need at least {total_required} videos in {video_dir}, found {len(ids)}."
        )

    for name, value in [
        ("train-videos", args.train_videos),
        ("val-videos", args.val_videos),
        ("test-videos", args.test_videos),
    ]:
        if value % 2 != 0:
            raise ValueError(f"{name} must be even, got {value}.")

    rng = random.Random(args.seed)
    rng.shuffle(ids)
    selected = ids[:total_required]
    train_ids, val_ids, test_ids = split_ids(
        selected, args.train_videos, args.val_videos, args.test_videos
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, split_ids_ in [
        ("train.json", train_ids),
        ("val.json", val_ids),
        ("test.json", test_ids),
    ]:
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(pair_ids(split_ids_), f, indent=2)

    print(
        "Wrote FaceForensics++ splits to "
        f"{output_dir} with train/val/test videos = "
        f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir",
        default="data/FaceForensics++/original_sequences/youtube/c23/videos",
    )
    parser.add_argument("--output-dir", default="data/FaceForensics++")
    parser.add_argument("--train-videos", type=int, default=800)
    parser.add_argument("--val-videos", type=int, default=100)
    parser.add_argument("--test-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
