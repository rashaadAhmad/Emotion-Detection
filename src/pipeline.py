import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

from .emotion_recognition_model import EmotionRecognition
from .face_detect_model import FaceDetect


class EmotionDetectionPipeline:
    """
    Pipeline for emotion detection on images.
    Combines face detection and emotion recognition.
    """
    
    # Emotion class labels (based on dataset structure - alphabetical order from ImageFolder)
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def __init__(
        self,
        face_detect_weights="checkpoints/face_detect/facedetection.pt",
        emotion_weights="checkpoints/emotion_recognition/best.pt",
        device=None
    ):
        """
        Initialize the emotion detection pipeline.
        
        Args:
            face_detect_weights: Path to face detection model weights
            emotion_weights: Path to emotion recognition model weights
            device: Device to run inference on ('cpu' or 'cuda'). Auto-detects if None.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize face detection model
        self.face_detector = FaceDetect(weights=face_detect_weights, device=str(self.device))
        
        # Initialize emotion recognition model
        self.emotion_model = EmotionRecognition()
        self.emotion_model.load_state_dict(
            torch.load(emotion_weights, map_location=self.device)
        )
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        # Define preprocessing transforms (same as training)
        # Note: No RandomHorizontalFlip for inference
        mean = [0.5074, 0.5074, 0.5074]
        std = [0.2121, 0.2121, 0.2121]
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),  # Match training input size
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            
        Returns:
            YOLO detection results
        """
        results = self.face_detector.detect(image)
        return results
    
    def preprocess_face(self, face_image):
        """
        Preprocess a face image for emotion recognition.
        
        Args:
            face_image: Face image (PIL Image or numpy array)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        tensor = self.transform(face_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image.
        
        Args:
            face_image: Preprocessed face image tensor or raw face image
            
        Returns:
            Dictionary with emotion label, confidence, and all probabilities
        """
        # Preprocess if not already a tensor
        if not isinstance(face_image, torch.Tensor):
            face_tensor = self.preprocess_face(face_image)
        else:
            face_tensor = face_image
        
        # Get prediction
        with torch.no_grad():
            outputs = self.emotion_model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion_idx = predicted.item()
            emotion_label = self.EMOTION_LABELS[emotion_idx]
            confidence_score = confidence.item()
            
            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()
            emotion_probs = {
                self.EMOTION_LABELS[i]: float(all_probs[i])
                for i in range(len(self.EMOTION_LABELS))
            }
        
        return {
            'emotion': emotion_label,
            'confidence': confidence_score,
            'probabilities': emotion_probs
        }
    
    def process_image(self, image, return_visualization=False):
        """
        Complete pipeline: detect faces and predict emotions.
        
        Args:
            image: Input image (numpy array, PIL Image, or path string)
            return_visualization: If True, returns annotated image
            
        Returns:
            List of dictionaries with face bounding boxes and emotions.
            If return_visualization=True, also returns annotated image.
        """
        # Load image if path string
        if isinstance(image, str):
            image = cv2.imread(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image_rgb = np.array(image)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        # Detect faces
        results = self.detect_faces(image_rgb)
        
        # Process each detected face
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract face region
                    face_region = image_rgb[y1:y2, x1:x2]
                    
                    if face_region.size == 0:
                        continue
                    
                    # Predict emotion
                    emotion_result = self.predict_emotion(face_region)
                    
                    # Store detection
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'emotion': emotion_result['emotion'],
                        'confidence': emotion_result['confidence'],
                        'probabilities': emotion_result['probabilities']
                    }
                    detections.append(detection)
                    
                    # Draw on image if visualization requested
                    if return_visualization:
                        # Draw bounding box
                        cv2.rectangle(
                            annotated_image, 
                            (x1, y1), (x2, y2), 
                            (0, 255, 0), 2
                        )
                        
                        # Draw emotion label
                        label = f"{emotion_result['emotion']}: {emotion_result['confidence']:.2f}"
                        label_size, _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            annotated_image,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (0, 255, 0), -1
                        )
                        cv2.putText(
                            annotated_image, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2
                        )
        
        if return_visualization:
            return detections, annotated_image
        return detections


# Convenience function for quick inference
def detect_emotions(
    image,
    face_detect_weights="checkpoints/face_detect/facedetection.pt",
    emotion_weights="checkpoints/emotion_recognition/best.pt",
    device=None,
    return_visualization=False
):
    """
    Convenience function to detect emotions in an image.
    
    Args:
        image: Input image (numpy array, PIL Image, or path)
        face_detect_weights: Path to face detection weights
        emotion_weights: Path to emotion recognition weights
        device: Device to use ('cpu' or 'cuda')
        return_visualization: If True, returns annotated image
        
    Returns:
        Detection results (and annotated image if return_visualization=True)
    """
    pipeline = EmotionDetectionPipeline(
        face_detect_weights=face_detect_weights,
        emotion_weights=emotion_weights,
        device=device
    )
    return pipeline.process_image(image, return_visualization=return_visualization)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Initialize pipeline
        pipeline = EmotionDetectionPipeline()
        
        # Process image
        detections, annotated_img = pipeline.process_image(
            image_path, 
            return_visualization=True
        )
        
        # Print results
        print(f"Found {len(detections)} face(s):")
        for i, det in enumerate(detections):
            print(f"\nFace {i+1}:")
            print(f"  Bounding box: {det['bbox']}")
            print(f"  Emotion: {det['emotion']} (confidence: {det['confidence']:.2%})")
            print(f"  All probabilities:")
            for emotion, prob in det['probabilities'].items():
                print(f"    {emotion}: {prob:.2%}")
        
        # Save annotated image
        output_path = "output_annotated.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        print("Usage: python pipeline.py <image_path>")

