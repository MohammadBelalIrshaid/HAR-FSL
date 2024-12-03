import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


# 2. Helper Functions to Load and Split Data
def load_videos_from_folder(folder_path, frame_size=(128, 128), max_frames=None):
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    frames = []

    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transforms.ToTensor()(frame_pil)
            frames.append(transform(frame_tensor))

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()

    return torch.stack(frames)


def split_data(video_data, labels, test_size=0.2):
    train_data, test_data, train_labels, test_labels = train_test_split(video_data, labels, test_size=test_size, random_state=42)

    # Convert train_labels and test_labels to tensors
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    
    print(f"train_data type: {type(train_data)}, shape: {train_data.shape if isinstance(train_data, torch.Tensor) else len(train_data)}")
    print(f"train_labels type: {type(train_labels)}, length: {len(train_labels)}")

    return train_data, test_data, train_labels, test_labels




def load_support_data(support_class_folders, frame_size=(128, 128), max_frames=None, test_size=0.2):
    video_tensors = []
    label_tensors = []
    for class_idx, class_folder in enumerate(support_class_folders):
        video_tensor = load_videos_from_folder(class_folder, frame_size, max_frames)
        video_tensors.extend(video_tensor)  # Flatten video tensors
        label_tensors.extend([class_idx] * len(video_tensor))

    video_tensors = torch.stack(video_tensors)  # Convert to single tensor
    
    print(video_tensors.shape)  # Should be [total_frames, channels, height, width]
    print(len(label_tensors))   # Should match total_frames

    return split_data(video_tensors, label_tensors, test_size)



def load_query_data(query_class_folders, frame_size=(128, 128), max_frames=None, test_size=0.2):
    return load_support_data(query_class_folders, frame_size, max_frames, test_size)


# 3. Prototypical Network
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


def calculate_accuracy(embeddings, labels, prototypes):
    distances = torch.cdist(embeddings, prototypes, p=2)
    _, predicted_labels = torch.min(distances, dim=1)
    accuracy = (predicted_labels == labels).float().mean()
    return accuracy.item()


# 4. Training and Fine-Tuning Functions
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
            loss = criterion(-distances, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy correctly: accumulate accuracy over the batch and then average
            batch_accuracy = calculate_accuracy(embeddings, labels, prototypes)
            epoch_accuracy += batch_accuracy * inputs.size(0)
            total_samples += inputs.size(0)

        # Average accuracy for the epoch
        epoch_accuracy /= total_samples

        train_losses.append(epoch_loss / len(support_loader))
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return train_losses, train_accuracies



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

            # Compute accuracy
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += accuracy * inputs.size(0)
            total_samples += inputs.size(0)

        # Store the average loss and accuracy for the epoch
        fine_tune_losses.append(epoch_loss / len(query_loader))
        fine_tune_accuracies.append(epoch_accuracy / total_samples)

        print(f"Fine-tune Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy / total_samples:.4f}")

    return fine_tune_losses, fine_tune_accuracies



# 5. Evaluation
def evaluate(model, test_loader, prototypes, num_samples=5):
    model.eval()
    total_accuracy = 0
    total_samples = 0
    images_to_display = []
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            embeddings = model(inputs)
            distances = torch.cdist(embeddings, prototypes, p=2)
            _, predicted_labels = torch.min(distances, dim=1)
            
            total_accuracy += (predicted_labels == labels).float().sum().item()
            total_samples += labels.size(0)
            
            # Collect images and predictions to display
            for i in range(min(num_samples, inputs.size(0))):
                images_to_display.append(inputs[i].cpu())
                predictions.append(predicted_labels[i].cpu().item())
                true_labels.append(labels[i].cpu().item())

    accuracy = total_accuracy / total_samples
    return accuracy, images_to_display, predictions, true_labels



def visualize_predictions(images, predictions, true_labels, num_samples=3):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Convert tensor to numpy for displaying
        ax.axis('off')
        ax.set_aspect('auto')  # Adjust the aspect ratio
        ax.set_title(f"Pred: {predictions[i]} \nTrue: {true_labels[i]}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to prevent cropping
    plt.show()


def visualize_correct_predictions(images, predictions, true_labels, num_samples=3):
    """Visualize only correct predictions."""
    # Convert predictions and true_labels to tensors if they are not already
    predictions = torch.tensor(predictions) if not isinstance(predictions, torch.Tensor) else predictions
    true_labels = torch.tensor(true_labels) if not isinstance(true_labels, torch.Tensor) else true_labels

    # Ensure predictions and true_labels are on CPU if they are on GPU
    predictions = predictions.cpu() if predictions.is_cuda else predictions
    true_labels = true_labels.cpu() if true_labels.is_cuda else true_labels

    # Get the indices where the prediction matches the true label
    correct_indices = (predictions == true_labels).nonzero(as_tuple=True)[0]
    
    # Limit the number of samples to display
    num_samples = min(num_samples, len(correct_indices))
    
    if num_samples == 0:
        print("No correct predictions to visualize.")
        return
    
    # Select the images where the prediction is correct
    selected_images = [images[idx] for idx in correct_indices[:num_samples]]
    selected_predictions = [predictions[idx] for idx in correct_indices[:num_samples]]
    selected_true_labels = [true_labels[idx] for idx in correct_indices[:num_samples]]
    
    # Plot the images with predictions and true labels
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(selected_images[i].permute(1, 2, 0).cpu().numpy())  # Convert tensor to numpy for displaying
        ax.axis('off')

        ax.set_title(f"Pred: {selected_predictions[i]} \nTrue: {selected_true_labels[i]}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to prevent cropping
    plt.show()
    
def visualize_correct_predictions_by_label(images, predictions, true_labels, class_names, num_samples_per_class=1):
    """Visualize exactly one correct prediction for each label/class."""
    # Convert predictions and true_labels to tensors if they are not already
    predictions = torch.tensor(predictions) if not isinstance(predictions, torch.Tensor) else predictions
    true_labels = torch.tensor(true_labels) if not isinstance(true_labels, torch.Tensor) else true_labels

    # Ensure predictions and true_labels are on CPU if they are on GPU
    predictions = predictions.cpu() if predictions.is_cuda else predictions
    true_labels = true_labels.cpu() if true_labels.is_cuda else true_labels

    # Get the indices where the prediction matches the true label
    correct_indices = (predictions == true_labels).nonzero(as_tuple=True)[0]

    # Dictionary to store correct samples for each class
    correct_samples_by_class = {label: [] for label in range(len(class_names))}

    # Collect correct predictions grouped by class
    for idx in correct_indices:
        class_label = true_labels[idx].item()
        if len(correct_samples_by_class[class_label]) < num_samples_per_class:
            correct_samples_by_class[class_label].append(idx)

    # Prepare data for visualization
    selected_images = []
    selected_predictions = []
    selected_true_labels = []

    for class_label, indices in correct_samples_by_class.items():
        for idx in indices:
            selected_images.append(images[idx])
            selected_predictions.append(predictions[idx].item())
            selected_true_labels.append(true_labels[idx].item())

    # Limit the number of subplots to the number of collected samples
    num_samples = len(selected_images)

    if num_samples == 0:
        print("No correct predictions to visualize.")
        return

    # Plot the images with predictions and true labels
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    # Ensure `axes` is always iterable
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(selected_images[i].permute(1, 2, 0).cpu().numpy())  # Convert tensor to numpy for displaying
        ax.axis('off')

        # Convert numerical class labels to their names using `class_names`
        pred_label = class_names[selected_predictions[i]]
        true_label = class_names[selected_true_labels[i]]
        ax.set_title(f"Pred: {pred_label} \nTrue: {true_label}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to prevent cropping
    plt.show()





# 6. Visualization
def plot_training_metrics(train_losses, train_accuracies, fine_tune_losses, fine_tune_accuracies):
    """Plot training metrics: losses and accuracies."""
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(fine_tune_losses, label='Fine-Tune Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    # Convert accuracies (if they are tensors) to numpy for plotting
    train_accuracies = [acc.item() if isinstance(acc, torch.Tensor) else acc for acc in train_accuracies]
    fine_tune_accuracies = [acc.item() if isinstance(acc, torch.Tensor) else acc for acc in fine_tune_accuracies]

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(fine_tune_accuracies, label='Fine-Tune Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
def filter_query_samples(video_data, labels, num_samples_per_class=10):
    """Filter a fixed number of samples per class."""
    filtered_indices = []
    class_sample_counts = {label: 0 for label in set(labels.tolist())}

    for idx, label in enumerate(labels.tolist()):
        if class_sample_counts[label] < num_samples_per_class:
            filtered_indices.append(idx)
            class_sample_counts[label] += 1

    # Filter video data and labels based on the selected indices
    filtered_video_data = video_data[filtered_indices]
    filtered_labels = labels[filtered_indices]

    return filtered_video_data, filtered_labels


# Updated Main Script
if __name__ == "__main__":
    # File paths
    support_class_folders = [
        r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Walking',
        r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Jumping'
    ]
    query_class_folders = [r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Boxing']

    frame_size = (128, 128)
    max_frames = 1500
    num_classes = 2

    # Load data
    support_train_videos, support_test_videos, support_train_labels, support_test_labels = load_support_data(
        support_class_folders, frame_size, max_frames
    )
    query_train_videos, query_test_videos, query_train_labels, query_test_labels = load_query_data(
        query_class_folders, frame_size, max_frames
    )

    # Apply filtering to query samples (optional)
    query_test_videos, query_test_labels = filter_query_samples(
        query_test_videos, query_test_labels, num_samples_per_class=10
    )

    # DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    support_train_loader = DataLoader(DVSDataset(support_train_videos, support_train_labels, transform), batch_size=16, shuffle=True)
    support_test_loader = DataLoader(DVSDataset(support_test_videos, support_test_labels, transform), batch_size=16, shuffle=False)
    query_train_loader = DataLoader(DVSDataset(query_train_videos, query_train_labels, transform), batch_size=16, shuffle=True)
    query_test_loader = DataLoader(DVSDataset(query_test_videos, query_test_labels, transform), batch_size=16, shuffle=False)

    # Model and training
    feature_extractor = resnet18(weights="IMAGENET1K_V1")
    feature_extractor.fc = nn.Identity()
    protonet = ProtoNet(feature_extractor).cuda()

    optimizer = optim.Adam(protonet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = train_protonet(protonet, support_train_loader, criterion, optimizer, num_classes)
    fine_tune_losses, fine_tune_accuracies = fine_tune(protonet, query_train_loader, criterion, optimizer)

    # Evaluate
    with torch.no_grad():
        support_embeddings = protonet(torch.stack([x[0] for x in support_train_loader.dataset]).cuda())
        prototypes = compute_prototypes(support_embeddings, support_train_labels.cuda(), num_classes)

    support_test_accuracy, _, _, _ = evaluate(protonet, support_test_loader, prototypes)
    print(f"Support Test Accuracy: {support_test_accuracy * 100:.2f}%")

    query_test_accuracy, images_to_display, predictions, true_labels = evaluate(protonet, query_test_loader, prototypes, num_samples=5)
    print(f"Query Test Accuracy: {query_test_accuracy * 100:.2f}%")

    # Visualize test predictions
    visualize_predictions(images_to_display, predictions, true_labels, num_samples=3)
    
    # Visualize correct predictions
    visualize_correct_predictions(images_to_display, predictions, true_labels, num_samples=3)
    
    class_names = ["Boxing", "Jumping", "Walking"]
    visualize_correct_predictions_by_label(images_to_display, predictions, true_labels, class_names, num_samples_per_class=1)

    # Visualize training metrics
    plot_training_metrics(train_losses, train_accuracies, fine_tune_losses, fine_tune_accuracies)

