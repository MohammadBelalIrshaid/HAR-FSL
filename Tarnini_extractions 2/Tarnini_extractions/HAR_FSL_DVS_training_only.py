import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 1. Custom Dataset for DVS Data
class DVSDataset(Dataset):
    def __init__(self, video_data, labels, transform=None):
        self.video_data = video_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.video_data[idx]
        label = self.labels[idx]

        sample = transforms.ToPILImage()(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Load video data from folder and preprocess
def load_videos_from_folder(folder_path, frame_size=(128, 128), max_frames=None):
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    frames = []

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize RGB values
    ])

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert NumPy array to PIL Image
            frame_pil = Image.fromarray(frame_rgb)

            # Convert PIL image to tensor
            frame_tensor = transforms.ToTensor()(frame_pil)

            # Apply normalization (already a tensor)
            frames.append(transform(frame_tensor))

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()

    return torch.stack(frames)


def load_support_data(support_class_folders, frame_size=(128, 128), max_frames=None):
    video_tensors = []
    label_tensors = []

    for class_idx, class_folder in enumerate(support_class_folders):
        video_tensor = load_videos_from_folder(class_folder, frame_size, max_frames)
        video_tensors.append(video_tensor)
        label_tensors.extend([class_idx] * video_tensor.shape[0])  # Assign labels for each frame

    return torch.cat(video_tensors), torch.tensor(label_tensors)


def load_query_data(query_class_folders, frame_size=(128, 128), max_frames=None):
    return load_support_data(query_class_folders, frame_size, max_frames)


# File paths for support and query set video folders
support_class_folders = [
    r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Walking',
    r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Boxing'
]  # Support classes

query_class_folders = [r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Jumping']  # Query class

# Load videos into tensors
frame_size = (128, 128)  # Resize frames to 128x128
max_frames = 2000  # Load up to 100 frames per video

# Support set
support_videos, support_labels = load_support_data(
    support_class_folders, frame_size, max_frames
)

# Query set
query_videos, query_labels = load_query_data(
    query_class_folders, frame_size, max_frames
)

print("Support Videos Shape:", support_videos.shape)
print("Support Labels Shape:", support_labels.shape)
print("Query Videos Shape:", query_videos.shape)
print("Query Labels Shape:", query_labels.shape)


# 2. Prototypical Network
class ProtoNet(nn.Module):
    def __init__(self, feature_extractor):
        super(ProtoNet, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        return self.feature_extractor(x)


def compute_prototypes(support_embeddings, support_labels, num_classes):
    prototypes = torch.zeros(num_classes, support_embeddings.size(1)).cuda()
    for label in range(num_classes):
        mask = support_labels == label
        class_embeddings = support_embeddings[mask]
        if class_embeddings.size(0) > 0:
            prototypes[label] = class_embeddings.mean(dim=0)
    return prototypes


# Helper function for calculating accuracy
def calculate_accuracy(embeddings, labels, prototypes):
    distances = torch.cdist(embeddings, prototypes, p=2)  # L2 distance
    _, predicted_labels = torch.min(distances, dim=1)  # Find the closest prototype
    accuracy = (predicted_labels == labels).float().mean()
    return accuracy.item()


# 3. Train Function with Metrics
def train_protonet(model, support_loader, criterion, optimizer, num_classes, epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        total_samples = 0

        for inputs, labels in support_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            embeddings = model(inputs)

            prototypes = compute_prototypes(embeddings, labels, num_classes)

            distances = torch.cdist(embeddings, prototypes, p=2)
            loss = criterion(-distances, labels)  # Use the negative of distances as logits

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(embeddings, labels, prototypes) * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = epoch_loss / len(support_loader)
        avg_accuracy = epoch_accuracy / total_samples

        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return train_losses, train_accuracies


# 4. Fine-Tuning Function with Accuracy Calculation
def fine_tune(model, query_loader, criterion, optimizer, epochs=5):
    model.train()
    fine_tune_losses = []
    fine_tune_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        total_samples = 0

        for inputs, labels in query_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += accuracy * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = epoch_loss / len(query_loader)
        avg_accuracy = epoch_accuracy / total_samples

        fine_tune_losses.append(avg_loss)
        fine_tune_accuracies.append(avg_accuracy)

        print(f"Fine-tune Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return fine_tune_losses, fine_tune_accuracies


# 5. Visualization Functions
def plot_training_metrics(train_losses, train_accuracies, fine_tune_losses, fine_tune_accuracies):
    plt.figure(figsize=(12, 5))

    # Move tensors to CPU for plotting (if they are on CUDA)
    train_losses = torch.tensor(train_losses).cpu().numpy()
    train_accuracies = torch.tensor(train_accuracies).cpu().numpy()
    fine_tune_losses = torch.tensor(fine_tune_losses).cpu().numpy()
    fine_tune_accuracies = torch.tensor(fine_tune_accuracies).cpu().numpy()

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(fine_tune_losses, label='Fine-Tune Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(fine_tune_accuracies, label='Fine-Tune Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# 6. Visualize Predictions on Query Data
def visualize_predictions(model, query_loader, prototypes, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(query_loader):
            if i >= num_samples:
                break
            inputs, labels = inputs.cuda(), labels.cuda()
            embeddings = model(inputs)
            distances = torch.cdist(embeddings, prototypes, p=2)
            _, predicted_labels = torch.min(distances, dim=1)

            # Show the image and prediction
            axes[i].imshow(inputs[0].cpu().permute(1, 2, 0))  # Show the first image in the batch
            axes[i].set_title(f"True: {labels[0].item()}, Pred: {predicted_labels[0].item()}")
            axes[i].axis('off')

    plt.show()


# 7. Main Block
if __name__ == "__main__":
    # Prepare data and models as before...
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    support_dataset = DVSDataset(support_videos, support_labels, transform=transform)
    query_dataset = DVSDataset(query_videos, query_labels, transform=transform)

    support_loader = DataLoader(support_dataset, batch_size=16, shuffle=True)
    query_loader = DataLoader(query_dataset, batch_size=16, shuffle=True)

    feature_extractor = resnet18(weights="IMAGENET1K_V1")
    feature_extractor.fc = nn.Identity()

    protonet = ProtoNet(feature_extractor).cuda()
    optimizer = optim.Adam(protonet.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train on support set
    train_losses, train_accuracies = train_protonet(protonet, support_loader, criterion, optimizer, num_classes=2, epochs=10)

    # Fine-tune on query set
    fine_tune_losses, fine_tune_accuracies = fine_tune(protonet, query_loader, criterion, optimizer, epochs=5)

    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, fine_tune_losses, fine_tune_accuracies)

    # Compute prototypes and visualize predictions
    with torch.no_grad():
        support_embeddings = protonet(support_videos.cuda())
        prototypes = compute_prototypes(support_embeddings, support_labels.cuda(), num_classes=2)
        visualize_predictions(protonet, query_loader, prototypes, num_samples=5)

    # Save model weights
    torch.save(protonet.state_dict(), "protonet_weights.pth")
