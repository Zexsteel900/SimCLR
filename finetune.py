import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_eurosat_finetune
from model import SimCLR, LinearClassifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained encoder
    model = SimCLR()
    model.encoder.load_state_dict(torch.load("checkpoints/simclr_encoder.pth"))
    model = model.to(device)

    classifier = LinearClassifier().to(device)

    full_model = nn.Sequential(model.encoder, classifier).to(device)

    train_loader, test_loader = get_eurosat_finetune(label_fraction=0.05)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(full_model.parameters(), lr=1e-4)

    # Training
    for epoch in range(20):
        full_model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = full_model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    full_model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = full_model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Fine-tuning Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()