import os
import threading
import time
from typing import Callable, Optional

from PIL import Image

from .failure_db import FailureDatabase
from .local_specialist_model import LocalSpecialistModel


class SpecialistTrainer:
    """
    Simple background trainer that watches corrections log and incrementally
    feeds new crops into the LocalSpecialistModel.
    """

    def __init__(
        self,
        database: FailureDatabase,
        specialist: LocalSpecialistModel,
        poll_interval_sec: float = 10.0,
        on_new_sample: Optional[Callable[[str, str], None]] = None,
    ):
        self.db = database
        self.specialist = specialist
        self.poll_interval_sec = poll_interval_sec
        self.on_new_sample = on_new_sample
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_size = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        corrections_path = self.db.corrections_log
        while not self._stop_event.is_set():
            try:
                if os.path.exists(corrections_path):
                    new_lines = self._read_new_lines(corrections_path)
                    for line in new_lines:
                        import json

                        data = json.loads(line)
                        crop_path = data.get("crop_image_path")
                        label = data.get("label")
                        if crop_path and label and os.path.exists(crop_path):
                            image = Image.open(crop_path).convert("RGB")
                            self.specialist.learn_from_correction(image, label)
                            if self.on_new_sample is not None:
                                self.on_new_sample(crop_path, label)
            except Exception:
                # Avoid crashing the background loop due to transient errors
                pass
            time.sleep(self.poll_interval_sec)

    def _read_new_lines(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if self._last_size >= len(lines):
            return []
        new = lines[self._last_size :]
        self._last_size = len(lines)
        return new


