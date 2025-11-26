## REFLECT Adaptation (Failure -> Correction -> Learning)

This module adds a dynamic “learn from failure” loop to REFLECT:

- Failure/Correction Database: logs failures and human corrections with images.
- Local Specialist Model: lightweight CLIP feature memory for environment-specific recognition.
- Decision Fusion: tries the specialist first; falls back to V-LLM; learns on failure.
- Training Scheduler: background process to ingest new corrections automatically.

### Quick Start

1) Provide two adapters from your existing REFLECT stack:

```python
from PIL import Image
from reflect.adaptation import create_adaptive_executor

def get_vllm_description(image: Image.Image) -> str:
    # TODO: plug into your existing V-LLM scene description
    return "no mug detected"

def reflect_llm_compare(task_command: str, scene_description: str):
    # TODO: plug into REFLECT success/diagnosis comparator.
    # Return (is_success: bool, failure_reason: str)
    if "mug" in scene_description:
        return True, ""
    return False, "V-LLM: I cannot see a mug."

executor = create_adaptive_executor(get_vllm_description, reflect_llm_compare)
```

2) Execute tasks with adaptation:

```python
image = Image.open("your_scene.png").convert("RGB")
ok = executor.execute_task_with_adaptation(task_command="mug", current_image=image)
```

On failure, a record is appended to `reflect/adaptation/data/`. Hook up your UI to supply a crop and label (see `human_in_loop.py`) so the specialist learns incrementally.

### Human-in-the-Loop

`human_in_loop.get_human_correction_blocking` is a stub. Replace with your UI that lets a user:
- Draw a bounding box on the failure image to yield a crop.
- Provide the correct label (e.g., "mug").

### Background Training

Optionally run a background trainer that watches for new corrections and feeds them to the specialist:

```python
from reflect.adaptation import FailureDatabase, LocalSpecialistModel, SpecialistTrainer

db = FailureDatabase()
sp = LocalSpecialistModel()
trainer = SpecialistTrainer(db, sp, poll_interval_sec=5.0)
trainer.start()
# ... later ...
trainer.stop()
```

### Persistence

Specialist memory persists to `reflect/adaptation/data/specialist_memory.pkl`. Failure and correction logs live under `reflect/adaptation/data/` as JSONL, and images under `images/`.


