import time 
import torch 
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model import build_model
from cd_dataset import CatsDogsDataset


def evaluate(model, loader, device):
    """
    Evaluate the model on a given DataLoader.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on (CPU or GPU).

    Returns:
        float: Average loss over the evaluation dataset.
    """
    model.eval()
    correct, total = 0,0 
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total 

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, processor = build_model()
    model.to(device)

    train_ds = CatsDogsDataset(csv_path="data/processed/train.csv", processor=processor)
    val_ds = CatsDogsDataset(csv_path="data/processed/val.csv", processor=processor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
    )

    history = {"train_loss": [], "val_accuracy": []}
    best_val_acc = 0.0


    for epoch in range(epochs):
        model.train()
        start = time.time()
        running_loss = 0.0 

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        history["train_loss"].append(avg_loss)
        history["val_accuracy"] .append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained("models/best_model")
            processor.save_pretrained("models/best_model")
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    return history


if __name__ == "__main__":
    train()