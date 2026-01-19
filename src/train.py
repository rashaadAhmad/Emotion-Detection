import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader
from tqdm import tqdm

from .emotion_recognition_model import EmotionRecognition
from .face_detect_model import FaceDetect
from .dataset import train_loader


def train_emotion_recognition(net:EmotionRecognition,train_loader ,num_epochs=2,device="cpu"):
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_list=[]
    accuracy_list=[]
    lowest_loss=float("inf")

    for epoch in range(num_epochs):
        total = 0
        correct = 0
        batch_count = 0
        epoch_loss = 0.0
        for i, (images, labels) in tqdm(enumerate(train_loader, 0),total=len(train_loader)):
            try:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()
                batch_count += 1
            except:break
        
        if epoch_loss<lowest_loss:
            lowest_loss=epoch_loss
            torch.save(net.state_dict(), "checkpoints/emotion_recognition/best.pt")

        avg_loss = epoch_loss / batch_count if batch_count else 0.0
        accuracy = (correct / total) if total else 0.0

        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_loss:.4f}, Accuracy: {100 * accuracy:.2f}%")

def train_face_detect(net:FaceDetect,data_path, num_epochs=2,device="cpu"):
    results = net.train(
        data=data_path,   # path to dataset config
        epochs=num_epochs,          # number of epochs
        imgsz=640,          # image size
        batch=16,           # batch size
        device=device            # GPU device (0 for first CUDA GPU)
    )
    return results

if __name__=="__main__":
    emotion=EmotionRecognition()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion.to(device)
    
    # Try to load checkpoint if it exists
    try:
        emotion.load_state_dict(torch.load("checkpoints/emotion_recognition/best.pt", map_location=device))
        print("Loaded existing checkpoint")
    except FileNotFoundError:
        print("No checkpoint found, starting training from scratch")
    
    train_emotion_recognition(emotion,train_loader,7,device)
