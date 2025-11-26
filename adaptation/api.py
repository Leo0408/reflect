from typing import Callable, Tuple

from PIL import Image

from .decision_fusion import AdaptiveExecutor, AdaptiveConfig
from .failure_db import FailureDatabase
from .local_specialist_model import LocalSpecialistModel


def create_adaptive_executor(
    get_vllm_description: Callable[[Image.Image], str],
    reflect_llm_compare: Callable[[str, str], Tuple[bool, str]],
    specialist_threshold: float = 0.8,
    db_root: str = "reflect/adaptation/data",
) -> AdaptiveExecutor:
    """
    Factory to instantiate AdaptiveExecutor with sensible defaults.
    The two callables are small adapters to your existing REFLECT code:
      - get_vllm_description(image) -> str
      - reflect_llm_compare(task_command, scene_description) -> (is_success, failure_reason)
    """
    config = AdaptiveConfig(specialist_threshold=specialist_threshold, db_root=db_root)
    specialist = LocalSpecialistModel(similarity_threshold=specialist_threshold)
    database = FailureDatabase(root_dir=db_root)
    return AdaptiveExecutor(
        get_vllm_description=get_vllm_description,
        reflect_llm_compare=reflect_llm_compare,
        config=config,
        specialist=specialist,
        database=database,
    )


