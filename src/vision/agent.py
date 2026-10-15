import cv2
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer
from ultralytics import YOLO

class EdgeVisionAgent:
    def __init__(self, vlm_model_id: str = "vikhyatk/moondream2"):
        print("Loading YOLOv8 detection model...")
        self.yolo = YOLO("yolov8n.pt")  # Lightweight nano model for fast edge detection

        print(f"Loading Quantized VLM ({vlm_model_id})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load VLM with half-precision (FP16) or INT4 quantization for low VRAM usage
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_model_id, trust_remote_code=True)
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            vlm_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map={"": self.device}
        )
        print(f"EdgeVisionAgent initialized successfully on device: {self.device}")

    def process_frame(self, frame, query: str = "Describe any potential security risks in this frame."):
        """Processes a single video frame using YOLO filter + conditional VLM invocation."""
        # 1. Run YOLOv8 object detection
        results = self.yolo(frame, verbose=False)[0]
        detected_objects = [self.yolo.names[int(box.cls[0])] for box in results.boxes]

        # 2. Define trigger criteria (e.g., trigger VLM only if person, backpack, or vehicle is detected)
        high_risk_classes = {"person", "backpack", "suitcase", "car", "knife"}
        trigger_vlm = any(obj in high_risk_classes for obj in detected_objects)

        vlm_description = None
        if trigger_vlm:
            # Convert OpenCV BGR frame to PIL Image for the VLM
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(image_rgb)
            
            # Encode image and query with Moondream2
            encoded_image = self.vlm_model.encode_image(pil_img)
            vlm_description = self.vlm_model.answer_question(encoded_image, query, self.vlm_tokenizer)

        return {
            "detected_objects": list(set(detected_objects)),
            "vlm_triggered": trigger_vlm,
            "visual_analysis": vlm_description if trigger_vlm else "Normal frame. No high-risk triggers."
        }