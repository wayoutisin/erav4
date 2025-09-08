import torch
from dataset import MNISTDataset
from model import MNISTModel
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("mnist")
train_data = dataset["train"]
test_data = dataset["test"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def train():
    train_dataset = MNISTDataset(train_data, transform)
    test_dataset = MNISTDataset(test_data, transform)

    # Create DataLoaders
    batch_size = 100
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )

    # Create model
    model = MNISTModel()
    count_parameters(model)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {int((i/100) + 1 if i > 0 else 1)}, Records {i*len(images)}, Loss: {loss.item()}")
        accuracy = evaluate_accuracy(model, test_loader)
        print(f"Accuracy on test data: {accuracy:.2f}%")

        # Break out of the loop if the accuracy reaches or goes above 95%
        if accuracy >= 95.0:
            break;

def count_parameters(model):
    total_params = 0
    print("\nModel Parameter Breakdown:")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
        print(f"Layer: {name:<20} | Parameters: {param}")
    print(f"\nTotal Trainable Parameters: {total_params}")
    return total_params

def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    train()
