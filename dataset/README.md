# Datasets

This project requires two separate datasets: one for face detection and one for emotion recognition. Both datasets are available on Kaggle and need to be downloaded and placed in the correct directories for the project to function.

## 1. Emotion Recognition Dataset

The emotion recognition model is trained on the **Human Face Emotions** dataset.

-   **Kaggle URL:** [https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

### Setup Instructions

1.  Navigate to the Kaggle URL above and download the dataset archive (`archive.zip`). You will need a Kaggle account to do this.
2.  The downloaded archive contains `train` and `test` directories.
3.  Extract the `train` and `test` directories into the `dataset/emotions/` directory. The final structure should match the one already present in the project:
    ```
    dataset/
    └── emotions/
        ├── train/
        │   ├── angry/
        │   ├── happy/
        │   └── ...
        └── test/
            ├── angry/
            ├── happy/
            └── ...
    ```

## 2. Face Detection Dataset (Optional)

The face detection model uses a pre-trained YOLO model, but if you wish to retrain it, you will need the **FDDB Subset for Face Detection** dataset in YOLO format.

-   **Kaggle URL:** [https://www.kaggle.com/datasets/tanmaypawar/fddb-subset-for-face-detection-yolo-format](https://www.kaggle.com/datasets/tanmaypawar/fddb-subset-for-face-detection-yolo-format)

### Setup Instructions

1.  Navigate to the Kaggle URL above and download the dataset archive.
2.  Create a new directory named `face` inside this `dataset` directory.
3.  Extract the contents of the downloaded archive into the `dataset/face/` directory. The `data.yaml` file, which is required for YOLO training, should be located at `dataset/face/data.yaml`.

After following these steps, your `dataset` directory will be correctly populated for both training and evaluation.
