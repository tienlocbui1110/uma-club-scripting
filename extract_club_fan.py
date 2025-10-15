#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))

import utils.opencv as opencv
from club_video_parsing import extract_video

def main():
    parser = argparse.ArgumentParser(description="Extract club member data from a club video (no bot).")
    parser.add_argument("video_path", help="Path to the input video file (e.g., uma.mp4)")
    parser.add_argument("-o", "--output", default="club_data.json", help="Output JSON file (default: club_data.json)")
    parser.add_argument("--fps", type=int, default=12, help="Target FPS to sample from the video (default: 12)")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"[ERROR] File not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Reading video: {args.video_path}")
    try:
        opencv.init_paddleocr()
        print(f"[INFO] Initialized Paddle ocr")

        data = extract_video(args.video_path, args.fps)
    except Exception as e:
        print("[ERROR] Failed to extract data:", e, file=sys.stderr)
        sys.exit(2)

    total = sum(len(g) for g in data) if isinstance(data, list) else 0
    print(f"[INFO] Extracted {total} entries.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved to {args.output}")

if __name__ == "__main__":
    main()
