import gradio as gr
import numpy as np
from PIL import Image
import cv2

from src.pipeline import EmotionDetectionPipeline

# Initialize the pipeline
pipeline = EmotionDetectionPipeline()

def emotion_detection(image):
    """
    Function to be used in the Gradio interface.
    Takes an image, processes it, and returns the annotated image.
    """
    # The pipeline expects a BGR image, but Gradio provides RGB
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process the image
    detections, annotated_image = pipeline.process_image(
        image_bgr, 
        return_visualization=True
    )
    
    # Convert the annotated image back to RGB for Gradio
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(annotated_image_rgb)

# Create the Gradio interface
iface = gr.Interface(
    fn=emotion_detection,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Image(type="pil", label="Annotated Image"),
    title="Emotion Detection",
    description="Upload an image to detect emotions on faces.",
    #allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
