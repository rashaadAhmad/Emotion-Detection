# Emotion Detection

This project is a real-time emotion detection application that uses a deep learning model to identify emotions from human faces in images. The application is built with Python and utilizes PyTorch for the model, OpenCV for image processing, and Gradio for the user interface.

## Features

- **Face Detection**: Automatically detects faces in an uploaded image.
- **Emotion Recognition**: Classifies the emotion of each detected face into one of seven categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- **Web Interface**: A simple and intuitive web interface powered by Gradio to upload images and view the results.

## Repository

- **GitHub:** [https://github.com/rashaadAhmad/Emotion-Detection](https://github.com/rashaadAhmad/Emotion-Detection)
- **Hugging Face:** [https://huggingface.co/RashaadAhmad/Emotion-Detection/](https://huggingface.co/RashaadAhmad/Emotion-Detection/)

## Project Structure

```
├── app.py                  # Main application file with the Gradio interface
├── requirements.txt        # Project dependencies
├── checkpoints/            # Contains pre-trained model weights
│   ├── emotion_recognition/
│   └── face_detect/
├── dataset/                # Image data for training and testing
│   └── emotions/
└── src/                    # Source code for the project
    ├── dataset.py              # Handles data loading
    ├── emotion_recognition_model.py # Emotion recognition model architecture
    ├── face_detect_model.py    # Face detection model architecture
    ├── pipeline.py             # Chains face detection and emotion recognition
    └── train.py                # Script for training the models
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rashaadAhmad/Emotion-Detection.git # or
    git clone https://huggingface.co/RashaadAhmad/Emotion-Detection
    cd emotion-detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

[Click Here](https://huggingface.co/spaces/RashaadAhmad/Emotion-Detection) to try it out.
To start the application and its web interface, run the following command:

```bash
python app.py
```

This will launch a local web server. Open the provided URL in your browser to access the application. You can then upload an image to see the emotion detection in action.

## Training

The model has been pre-trained, and the weights are available in the `checkpoints` directory. However, if you wish to retrain the model on a different dataset, you can use the `train.py` script:

```bash
python src/train.py
```

Make sure your dataset is structured correctly in the `dataset` directory.
