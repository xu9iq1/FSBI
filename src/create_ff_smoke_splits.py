import argparse
import json
from pathlib import Path


def pair_ids(video_ids):
    return [video_ids[i : i + 2] for i in range(0, len(video_ids), 2)]


def main(args):
    video_dir = Path(args.video_dir)
    ids = sorted(path.stem for path in video_dir.glob("*.mp4"))

    required = args.train_videos + args.val_videos
    if len(ids) < required:
        raise ValueError(
            f"Need at least {required} videos in {video_dir}, found {len(ids)}."
        )
    if args.train_videos % 2 != 0 or args.val_videos % 2 != 0:
        raise ValueError("train-videos and val-videos must both be even.")

    selected = ids[:required]
    train_ids = selected[: args.train_videos]
    val_ids = selected[args.train_videos : required]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(pair_ids(train_ids), f, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(pair_ids(val_ids), f, indent=2)

    if args.write_test:
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(pair_ids(val_ids), f, indent=2)

    print(
        f"Wrote train.json with {len(train_ids)} videos and "
        f"val.json with {len(val_ids)} videos to {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir",
        default="data/FaceForensics++/original_sequences/youtube/c23/videos",
    )
    parser.add_argument("--output-dir", default="data/FaceForensics++")
    parser.add_argument("--train-videos", type=int, default=6)
    parser.add_argument("--val-videos", type=int, default=2)
    parser.add_argument("--write-test", action="store_true")
    main(parser.parse_args())
