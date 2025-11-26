#!/usr/bin/env python3
import argparse
import json
import os
from typing import Optional

from PIL import Image

from ..failure_db import FailureDatabase


def parse_bbox(bbox_str: Optional[str]):
    if bbox_str is None:
        return None
    parts = bbox_str.split(",")
    if len(parts) != 4:
        raise ValueError("bbox must be 'x1,y1,x2,y2'")
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Submit a human correction for a failure image.")
    parser.add_argument("--task", required=True, help='Task command, e.g., "mug" or "find mug"')
    parser.add_argument("--image", required=True, help="Path to the scene image to crop from")
    parser.add_argument("--label", required=True, help="Correct label for the crop, e.g., 'mug'")
    parser.add_argument("--bbox", required=False, help="Crop bbox as 'x1,y1,x2,y2'")
    parser.add_argument("--db-root", default="reflect/adaptation/data", help="Database root directory")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    image = Image.open(args.image).convert("RGB")
    crop = image
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = image.crop((x1, y1, x2, y2))

    db = FailureDatabase(root_dir=args.db_root)
    rec = db.log_correction(task_command=args.task, label=args.label, crop_image=crop, bbox_xyxy=bbox)
    out = {
        "message": "correction_submitted",
        "task": args.task,
        "label": args.label,
        "bbox": bbox,
        "crop_image_path": rec.crop_image_path,
        "db_root": os.path.abspath(args.db_root),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()


