import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image

try:
    import clip  # from OpenAI CLIP; ensure installed or locally available
except Exception:  # pragma: no cover
    clip = None


@dataclass
class LocalDetectionResult:
    label: str
    confidence: float
    # Optional: could include bbox and other metadata in future


class LocalSpecialistModel:
    """
    CLIP feature memory as a lightweight local specialist.
    - memory: dict[label] -> list[feature_tensor]
    - supports incremental learning from corrected crops
    - provides detect() returning high-confidence hits
    """

    def __init__(
        self,
        device: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        similarity_threshold: float = 0.8,
        persist_path: str = "reflect/adaptation/data/specialist_memory.pkl",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if clip is None:
            raise ImportError("CLIP is not available. Please install/open the CLIP package.")
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device, jit=False)
        self.model.eval()
        self.similarity_threshold = similarity_threshold
        self.persist_path = os.path.abspath(persist_path)

        self.memory: Dict[str, List[torch.Tensor]] = {}
        self._maybe_load()

    def _maybe_load(self) -> None:
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            # Tensors need to be moved to current device
            self.memory = {
                label: [feat.to(self.device) for feat in feats]
                for label, feats in data.get("memory", {}).items()
            }

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        cpu_memory = {k: [t.detach().cpu() for t in v] for k, v in self.memory.items()}
        with open(self.persist_path, "wb") as f:
            pickle.dump({"memory": cpu_memory}, f)

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat  # [1, d]

    @torch.no_grad()
    def learn_from_correction(self, image_crop: Image.Image, label: str) -> None:
        feature_vector = self._encode_image(image_crop)  # [1, d]
        if label not in self.memory:
            self.memory[label] = []
        self.memory[label].append(feature_vector.squeeze(0))
        self._persist()

    @torch.no_grad()
    def detect(self, current_image: Image.Image, label_to_find: str) -> Optional[LocalDetectionResult]:
        if label_to_find not in self.memory or len(self.memory[label_to_find]) == 0:
            return None
        query_features = self._encode_image(current_image)  # [1, d]
        query = query_features.squeeze(0)  # [d]
        best_sim = -1.0
        for expert_feat in self.memory[label_to_find]:
            sim = torch.matmul(query, expert_feat).item()
            if sim > best_sim:
                best_sim = sim
        if best_sim >= self.similarity_threshold:
            return LocalDetectionResult(label=label_to_find, confidence=float(best_sim))
        return None


