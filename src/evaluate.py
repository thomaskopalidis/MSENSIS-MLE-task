import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt
import seaborn as sns

from cd_dataset import CatsDogsDataset

MODEL_PATH = "models/best_model"
TEST_CSV = "data/processed/test.csv"
BATCH_SIZE = 32
CM_OUTPUT_PATH = "confusion_matrix.png"

"""
Standalone test-set evaluation for the already-trained Cats & Dogs model.

Loads the saved checkpoint in models/best_model/ and reports accuracy,
precision, recall, F1, and the confusion matrix on data/processed/test.csv.
"""
def evaluate_full(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, class_names=("Dog", "Not Dog"), output_path=CM_OUTPUT_PATH):
    """
    Plots the confusion matrix as a table (heatmap) with the labels
    'Dog' / 'Not Dog', similar to the screenshot you showed, and saves it
    to an image file.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="PiYG",           
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16, "weight": "bold"},
        linewidths=1,
        linecolor="white",
        ax=ax,
    )

    ax.set_title("Confusion Matrix - Output", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved as image to: {output_path}")


def print_metrics_table(metrics):
    """
    Prints the metrics in a table format to the console, similar to the second image
    (Accuracy / Precision / Recall / F1 score).
    """
    rows = [
        ("Accuracy", metrics["accuracy"] * 100),
        ("Precision", metrics["precision"] * 100),
        ("Recall", metrics["recall"] * 100),
        ("F1 score", metrics["f1"] * 100),
    ]

    label_width = max(len(name) for name, _ in rows) + 2
    print("\n" + "=" * (label_width + 12))
    print(f"{'Metric':<{label_width}}{'Value (%)':>10}")
    print("-" * (label_width + 12))
    for name, value in rows:
        print(f"{name:<{label_width}}{value:>10.2f}")
    print("=" * (label_width + 12))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {MODEL_PATH}...")
    model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    test_ds = CatsDogsDataset(csv_path=TEST_CSV, processor=processor)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(test_ds)} test images...")
    metrics = evaluate_full(model, test_loader, device)

    print("\n=== Test set results ===")
    print_metrics_table(metrics)

    print("\nConfusion matrix (rows=true, cols=pred, order=[cat, dog]):")
    print(f"  {metrics['confusion_matrix']}")

    # Plot and save the confusion matrix as an image
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=("Dog", "Not Dog"))


if __name__ == "__main__":
    main()
    print("\n" + "=" * (label_width + 12))
    print(f"{'Metric':<{label_width}}{'Value (%)':>10}")
    print("-" * (label_width + 12))
    for name, value in rows:
        print(f"{name:<{label_width}}{value:>10.2f}")
    print("=" * (label_width + 12))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {MODEL_PATH}...")
    model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    test_ds = CatsDogsDataset(csv_path=TEST_CSV, processor=processor)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(test_ds)} test images...")
    metrics = evaluate_full(model, test_loader, device)

    print("\n=== Test set results ===")
    print_metrics_table(metrics)

    print("\nConfusion matrix (rows=true, cols=pred, order=[cat, dog]):")
    print(f"  {metrics['confusion_matrix']}")

    # Plot and save the confusion matrix as an image
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=("Dog", "Not Dog"))


if __name__ == "__main__":
    main()