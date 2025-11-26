import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import glob
import sys

# --- 尝试导入 Reflect 库 ---
# IMPORTANT: This script relies on the 'reflect' library.
# Please ensure you have cloned the Reflect GitHub repository and installed it
# (e.g., by running 'pip install -e .' in the cloned 'reflect' directory).
# If the library is not found, a mock ReflectModel will be used, which will NOT
# perform actual Reflect inference and will give random failure scores.
_REFLECT_LIB_AVAILABLE = False
try:
    # Attempt to add the reflect repository to sys.path if it's a common setup
    # You might need to adjust this path or install Reflect as a package
    # Example for common setup:
    # if os.path.exists("./reflect"): # Assuming 'reflect' folder is in current directory
    #    sys.path.insert(0, os.path.abspath("./reflect"))
    
    from reflect.models import ReflectModel
    from reflect.config import get_reflect_cfg
    _REFLECT_LIB_AVAILABLE = True
    print("Reflect library (reflect.models.ReflectModel) imported successfully.")
except ImportError:
    print("Warning: Reflect library (reflect.models.ReflectModel) not found.")
    print("Please ensure the 'reflect' library is installed or available in your Python path.")
    print("Without it, the core Reflect model functionality will be mocked, leading to incorrect results.")
    print("Refer to https://github.com/real-stanford/reflect for installation instructions (e.g., 'pip install -e .').")

# --- Detectron2 相关的导入 ---
# Ensure you have detectron2 installed for mask prediction.
# Refer to https://detectron2.readthedocs.io/en/latest/install.html for installation.
_DETECTRON2_AVAILABLE = False
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    _DETECTRON2_AVAILABLE = True
except ImportError:
    print("Warning: Detectron2 not found. Mask generation will be mocked. This will prevent actual Reflect inference.")
    print("Please install Detectron2 to enable proper mask generation for Reflect model.")
    # Mock classes for Detectron2 if not available
    class MockCfg:
        def merge_from_file(self, *args, **kwargs): pass
        def set_resolution(self, *args, **kwargs): pass
        def freeze(self, *args, **kwargs): pass
        # Mimic attributes that might be accessed
        MODEL = type('MODEL', (object,), {'ROI_HEADS': type('ROI_HEADS', (object,), {'SCORE_THRESH_TEST': 0.5}), 'WEIGHTS': '', 'DEVICE': 'cpu'})
        INPUT = type('INPUT', (object,), {'MIN_SIZE_TEST': 800, 'MAX_SIZE_TEST': 1333})
    class MockInstances:
        def __init__(self):
            # Provide dummy data for testing the flow without actual detection
            self.pred_masks = torch.rand(1, 256, 256) > 0.5 # Example mask
            self.pred_boxes = torch.tensor([[0,0,256,256]]) # Example bbox
        def to(self, device): return self # Mock to method
        def __len__(self): return 1 # Always return one instance
        def area(self): return torch.tensor([1000.0]) # Dummy area
    class MockDefaultPredictor:
        def __init__(self, cfg): pass
        def __call__(self, img): return {"instances": MockInstances()}
    model_zoo = type('model_zoo', (object,), {'get_checkpoint_url': lambda x: ""}) # Mock function
    get_cfg = lambda: MockCfg()
    DefaultPredictor = MockDefaultPredictor

# --- Mock ReflectModel if Reflect library is not available ---
if not _REFLECT_LIB_AVAILABLE:
    class ReflectModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            print("Using MockReflectModel. This will NOT perform actual Reflect inference.")
        def forward(self, image, mask):
            # Simulate a random failure score for demonstration purposes
            # Reflect scores are typically between -1 (good) and 1 (failure)
            return {'failure_score': torch.tensor(np.random.rand() * 2 - 1).float()}

    class MockReflectConfig:
        def __init__(self):
            self.MODEL = type('MODEL', (object,), {'RESOLUTION': (1024, 1024)})
            self.FREEZE = type('FREEZE', (object,), {'BACKBONE': False}) # Reflect config specific
            self.merge_from_file = lambda x: None # Mock method
            self.freeze = lambda: None # Mock method
        def __call__(self): return self # Make it callable like get_reflect_cfg
    get_reflect_cfg = MockReflectConfig() # Assign the callable mock

class ReflectPredictor:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device for inference: {self.device}")

        # --- Reflect Model 加载 ---
        # 实例化 Reflect 模型并加载权重
        # The ReflectModel expects a config object.
        # This cfg should be consistent with how the model was trained.
        # get_reflect_cfg() is provided by the reflect library.
        # If Reflect library is not available, a mock cfg is used.
        self.reflect_cfg = get_reflect_cfg()
        # You might need to merge a specific reflect config file here if provided by the library.
        # E.g., self.reflect_cfg.merge_from_file("path/to/reflect_default_config.yaml")
        # Ensure the model resolution matches what ReflectModel expects
        self.resolution = self.reflect_cfg.MODEL.RESOLUTION # E.g., (1024, 1024)

        self.reflect_model = ReflectModel(self.reflect_cfg).to(self.device)
        try:
            # Load the state_dict from the provided .pth file
            # demo.ipynb loads 'model_state_dict' from the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.reflect_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Reflect model weights loaded from: {model_path}")
            else:
                self.reflect_model.load_state_dict(checkpoint) # In case .pth is just state_dict
                print(f"Reflect model weights (direct state_dict) loaded from: {model_path}")

            self.reflect_model.eval() # Set model to evaluation mode
        except Exception as e:
            print(f"Error loading Reflect model weights from {model_path}: {e}")
            if not _REFLECT_LIB_AVAILABLE:
                print("Using mock ReflectModel, no actual weights loaded.")
            else:
                print("Reflect model might not function correctly without proper weights.")
                # If real ReflectModel is there but loading fails, keep it in eval mode.
                self.reflect_model.eval()


        # --- Mask R-CNN Predictor 加载 (用于获取 mask) ---
        if _DETECTRON2_AVAILABLE:
            self.mask_predictor = self._load_mask_rcnn_predictor()
            print("Detectron2 Mask R-CNN predictor initialized.")
        else:
            self.mask_predictor = MockDefaultPredictor(get_cfg()) # Use mock if Detectron2 not available
            print("Using mock Mask R-CNN predictor. No actual masks will be generated.")


        # --- 图像预处理 transforms ---
        self.transform = T.Compose([
            T.Resize(self.resolution), # Resize to model expected resolution
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_mask_rcnn_predictor(self):
        """Loads a pre-trained Mask R-CNN model from Detectron2 model zoo."""
        cfg = get_cfg()
        # Use a common Mask R-CNN configuration
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for detections
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)
        return predictor

    def get_mask_from_image(self, image_pil):
        """
        Uses Detectron2 to get a segmentation mask for the primary object.
        Returns the mask (numpy array) and bounding box.
        """
        if not _DETECTRON2_AVAILABLE:
            # If Detectron2 is not available, simulate a mask
            mock_mask = np.ones(self.resolution, dtype=np.bool_)
            mock_bbox = np.array([0, 0, self.resolution[1], self.resolution[0]]) # [x1, y1, x2, y2]
            return mock_mask, mock_bbox

        img_np = np.array(image_pil.convert("BGR")) # Detectron2 expects BGR
        outputs = self.mask_predictor(img_np)
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            # As in demo.ipynb, we want the mask of the largest instance (by area)
            areas = instances.pred_boxes.area()
            idx = torch.argmax(areas)
            mask = instances.pred_masks[idx].squeeze().numpy() # Boolean mask
            bbox = instances.pred_boxes[idx].tensor.squeeze().numpy()
            return mask, bbox
        return None, None # No object detected

    @torch.no_grad()
    def predict_failure(self, image_path):
        """
        Predicts the failure score for a single image.
        Returns (failure_score, image_path) or (None, None) if processing fails.
        """
        if not _REFLECT_LIB_AVAILABLE:
            print(f"Warning: Reflect library not available. Returning random failure score for {image_path}.")
            return np.random.rand() * 2 - 1, image_path # Return random score if mock is used

        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None # Indicate failure to load

        # Get the segmentation mask
        mask_np, bbox_np = self.get_mask_from_image(image_pil)
        if mask_np is None:
            # print(f"No object mask detected for {os.path.basename(image_path)}. Skipping failure prediction.")
            return None, None # No mask means we can't run Reflect model

        # Prepare image tensor
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # Prepare mask tensor (should be same resolution as img_tensor, 1 channel)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        # Ensure mask is resized to the model's expected resolution if it's not already
        if mask_tensor.shape[2:] != img_tensor.shape[2:]:
            mask_tensor = T.Resize(img_tensor.shape[2:])(mask_tensor)

        try:
            # --- Reflect Model 的核心推理步骤 ---
            # Pass image and mask to the ReflectModel's forward method
            outputs = self.reflect_model(img_tensor, mask_tensor)
            failure_score = outputs['failure_score'].item() # Extract the failure score
            return failure_score, image_path
        except Exception as e:
            print(f"Error during Reflect model inference for {image_path}: {e}")
            return None, None # Indicate inference failure


def main(args):
    # Initialize the ReflectPredictor
    reflect_predictor = ReflectPredictor(args.model_path)

    # Get all image files from the specified directory
    image_paths = []
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    
    if not image_paths:
        print(f"Error: No supported image files found in '{args.image_dir}'. "
              f"Supported types: {', '.join(supported_extensions)}")
        return

    print(f"\nFound {len(image_paths)} images to process in '{args.image_dir}'.")

    failure_samples = []
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}...")
        failure_score, original_path = reflect_predictor.predict_failure(img_path)

        if failure_score is not None:
            if failure_score > args.threshold:
                print(f"  --> Identified as FAILURE (Score: {failure_score:.4f})")
                failure_samples.append({
                    "image_path": original_path,
                    "failure_score": failure_score
                })
            else:
                print(f"  --> Identified as SUCCESS (Score: {failure_score:.4f})")
        else:
            print(f"  --> Skipped {os.path.basename(img_path)} due to processing error (e.g., failed to load image or no mask detected).")

    print("\n--- Detected Failure Samples ---")
    if failure_samples:
        for sample in failure_samples:
            print(f"Image: {sample['image_path']}, Failure Score: {sample['failure_score']:.4f}")
    else:
        print("No failure samples detected in the dataset based on the threshold.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Reflect model inference on a dataset to detect potential failures in images."
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the directory containing input images (e.g., JPG, PNG).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Reflect model weights file (e.g., reflect_resnet50.pth).")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Failure score threshold. Images with a score > threshold are considered failures (default: 0.0).")
    
    args = parser.parse_args()
    main(args)