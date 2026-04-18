import argparse
import json
from pathlib import Path


def load_split_ids(split_path):
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return [video_id for pair in data for video_id in pair]


def count_files(path, suffix):
    if not path.exists():
        return 0
    return len(list(path.glob(f"*{suffix}")))


def main(args):
    base = Path(args.base_dir)
    c23_dir = base / "original_sequences" / "youtube" / args.comp

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_path = base / f"{split_name}.json"
        splits[split_name] = load_split_ids(split_path)

    print(f"Dataset root: {c23_dir}")
    print(f"Videos available: {count_files(c23_dir / 'videos', '.mp4')}")
    print()

    total_complete = 0
    for split_name, video_ids in splits.items():
        complete = 0
        missing = []
        for video_id in video_ids:
            frame_dir = c23_dir / "frames" / video_id
            landmark_dir = c23_dir / "landmarks" / video_id
            retina_dir = c23_dir / "retina" / video_id

            frame_count = count_files(frame_dir, ".png")
            landmark_count = count_files(landmark_dir, ".npy")
            retina_count = count_files(retina_dir, ".npy")
            if (
                frame_count >= args.num_frames
                and landmark_count >= args.num_frames
                and retina_count >= args.num_frames
            ):
                complete += 1
            else:
                missing.append(
                    f"{video_id}(f={frame_count},lm={landmark_count},rt={retina_count})"
                )

        total_complete += complete
        print(
            f"{split_name}: complete={complete}/{len(video_ids)} "
            f"missing={len(video_ids) - complete}"
        )
        if missing and args.show_missing:
            print("  " + ", ".join(missing[: args.max_missing]))

    print()
    print(f"Total split videos fully preprocessed: {total_complete}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="data/FaceForensics++")
    parser.add_argument("--comp", default="c23")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--show-missing", action="store_true")
    parser.add_argument("--max-missing", type=int, default=20)
    main(parser.parse_args())
