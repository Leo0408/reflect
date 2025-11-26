import dataclasses
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from PIL import Image


@dataclass
class FailureRecord:
    timestamp: float
    task_command: str
    failure_reason: str
    scene_image_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class CorrectionRecord:
    timestamp: float
    task_command: str
    label: str
    crop_image_path: Optional[str] = None
    bbox_xyxy: Optional[list] = None  # [x1, y1, x2, y2]
    meta: Optional[Dict[str, Any]] = None


class FailureDatabase:
    """
    A simple filesystem-backed database for logging failures and human corrections.
    - Stores images under {root}/images/
    - Appends JSONL entries under {root}/failures.jsonl and {root}/corrections.jsonl
    """

    def __init__(self, root_dir: str = "reflect/adaptation/data"):
        self.root_dir = os.path.abspath(root_dir)
        self.images_dir = os.path.join(self.root_dir, "images")
        self.failures_log = os.path.join(self.root_dir, "failures.jsonl")
        self.corrections_log = os.path.join(self.root_dir, "corrections.jsonl")
        self._lock = threading.Lock()
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.root_dir, exist_ok=True)

    def _save_image(self, image: Image.Image, prefix: str) -> str:
        ts = int(time.time() * 1000)
        filename = f"{prefix}_{ts}.png"
        path = os.path.join(self.images_dir, filename)
        image.save(path)
        return path

    def log_failure(
        self,
        task_command: str,
        failure_reason: str,
        scene_image: Optional[Image.Image] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        scene_image_path = None
        if scene_image is not None:
            scene_image_path = self._save_image(scene_image, "failure_scene")
        rec = FailureRecord(
            timestamp=time.time(),
            task_command=task_command,
            failure_reason=failure_reason,
            scene_image_path=scene_image_path,
            meta=meta,
        )
        self._append_jsonl(self.failures_log, asdict(rec))
        return rec

    def log_correction(
        self,
        task_command: str,
        label: str,
        crop_image: Optional[Image.Image] = None,
        bbox_xyxy: Optional[list] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CorrectionRecord:
        crop_image_path = None
        if crop_image is not None:
            crop_image_path = self._save_image(crop_image, "correction_crop")
        rec = CorrectionRecord(
            timestamp=time.time(),
            task_command=task_command,
            label=label,
            crop_image_path=crop_image_path,
            bbox_xyxy=bbox_xyxy,
            meta=meta,
        )
        self._append_jsonl(self.corrections_log, asdict(rec))
        return rec

    def _append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


