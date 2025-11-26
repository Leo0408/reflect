from typing import Optional, Tuple

from PIL import Image


def get_human_correction_blocking(
    failed_image: Image.Image,
    task_command: str,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Placeholder for human-in-the-loop correction.
    In production, replace with a UI that returns a crop and label.
    Here, returns (None, None) to indicate no correction provided.
    """
    # Integrate with a UI later (e.g., Gradio/Streamlit) that allows:
    # - drawing bbox to produce crop
    # - entering a label string
    return None, None


