from ultralytics import YOLO

class FaceDetect:
    def __init__(self, weights="checkpoints/face_detect/facedetection.pt",device="cpu"):
        self.model = YOLO(weights)
        self.device=device

    def detect(self, image):
        results=self.model(image)
        return results