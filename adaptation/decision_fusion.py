from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image

from .failure_db import FailureDatabase
from .human_in_loop import get_human_correction_blocking
from .local_specialist_model import LocalSpecialistModel


@dataclass
class AdaptiveConfig:
    specialist_threshold: float = 0.8
    db_root: str = "reflect/adaptation/data"


class AdaptiveExecutor:
    """
    Decision fusion for REFLECT + Local Specialist.
    This orchestrates:
    - try local specialist (high-confidence shortcut)
    - fallback to provided VLLM and REFLECT comparators
    - on failure, log + request correction + learn
    """

    def __init__(
        self,
        get_vllm_description: Callable[[Image.Image], str],
        reflect_llm_compare: Callable[[str, str], Tuple[bool, str]],
        config: Optional[AdaptiveConfig] = None,
        specialist: Optional[LocalSpecialistModel] = None,
        database: Optional[FailureDatabase] = None,
    ):
        self.config = config or AdaptiveConfig()
        self.specialist = specialist or LocalSpecialistModel(similarity_threshold=self.config.specialist_threshold)
        self.db = database or FailureDatabase(root_dir=self.config.db_root)
        self.get_vllm_description = get_vllm_description
        self.reflect_llm_compare = reflect_llm_compare

    def execute_task_with_adaptation(self, task_command: str, current_image: Image.Image) -> bool:
        # Ask Specialist first
        label_to_find = task_command  # assume task_command like "mug" or "find mug" handled upstream
        result = self.specialist.detect(current_image, label_to_find)

        if result is not None:
            scene_description = f"{result.label} detected with confidence {result.confidence:.3f}"
        else:
            # fallback to generic VLLM
            scene_description = self.get_vllm_description(current_image)

        is_success, failure_reason = self.reflect_llm_compare(task_command, scene_description)
        if is_success:
            return True

        # Failure path: log, request human correction, learn, log correction
        self.db.log_failure(task_command=task_command, failure_reason=failure_reason, scene_image=current_image)
        crop, correct_label = get_human_correction_blocking(current_image, task_command)
        if crop is not None and correct_label is not None:
            self.specialist.learn_from_correction(crop, correct_label)
            self.db.log_correction(task_command=task_command, label=correct_label, crop_image=crop)
        return False


