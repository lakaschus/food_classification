import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import random
from collections import defaultdict
from tqdm import tqdm  # Added tqdm for progress bars

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # ~66M parameters
MAX_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
DATA_PATH = "./data/food2ex_curated_full.csv"  # Update path if needed
OUTPUT_DIR = "./model_output"
EVAL_STEPS = 500
SAVE_STEPS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load and preprocess the data
def load_and_preprocess_data(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Extract features and labels
    X = df["BASETERM_NAME"].tolist()

    # Parse FACETS column to extract baseterm and facets
    baseterm_codes = []
    facet_codes = []

    # Add progress bar for data processing
    for facets_str in tqdm(df["FACETS"], desc="Processing data"):
        # Split into baseterm and facets
        if "#" in facets_str:
            baseterm, facets = facets_str.split("#", 1)
            baseterm_codes.append(baseterm)

            # Extract facet codes
            facet_list = []
            if facets:
                for facet in facets.split("$"):
                    if "." in facet:
                        facet_code = facet.split(".", 1)[
                            0
                        ]  # Take the part before the dot
                        facet_exp = facet
                        facet_list.append(facet_exp)

            facet_codes.append(facet_list)
        else:
            baseterm_codes.append(facets_str)
            facet_codes.append([])

    print(f"Loaded {len(X)} samples")
    return X, baseterm_codes, facet_codes


# Create label encoders
def create_label_encoders(baseterm_codes, facet_codes):
    # Create baseterm encoder
    unique_baseterms = list(set(baseterm_codes))
    baseterm_to_id = {code: idx for idx, code in enumerate(unique_baseterms)}
    id_to_baseterm = {idx: code for code, idx in baseterm_to_id.items()}

    # Create facet encoder
    # Flatten the list of facet codes
    all_facets = []
    for facet_list in facet_codes:
        all_facets.extend(facet_list)

    unique_facets = list(set(all_facets))
    facet_to_id = {code: idx for idx, code in enumerate(unique_facets)}
    id_to_facet = {idx: code for code, idx in facet_to_id.items()}

    print(f"Number of unique base terms: {len(unique_baseterms)}")
    print(f"Number of unique facet expressions: {len(unique_facets)}")

    return baseterm_to_id, id_to_baseterm, facet_to_id, id_to_facet


# Custom dataset class
class FoodClassificationDataset(Dataset):
    def __init__(
        self,
        texts,
        baseterm_codes,
        facet_codes,
        baseterm_to_id,
        facet_to_id,
        tokenizer,
        max_length=128,
    ):
        self.texts = texts
        self.baseterm_codes = baseterm_codes
        self.facet_codes = facet_codes
        self.baseterm_to_id = baseterm_to_id
        self.facet_to_id = facet_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        baseterm = self.baseterm_codes[idx]
        facets = self.facet_codes[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Create labels
        baseterm_label = self.baseterm_to_id[baseterm]

        # Create multi-hot encoding for facets
        facet_labels = torch.zeros(len(self.facet_to_id))
        for facet in facets:
            if facet in self.facet_to_id:
                facet_labels[self.facet_to_id[facet]] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "baseterm_label": baseterm_label,
            "facet_labels": facet_labels,
        }


# Custom model for multi-task learning
class FoodClassificationModel(torch.nn.Module):
    def __init__(self, model_name, num_baseterm_labels, num_facet_labels):
        super(FoodClassificationModel, self).__init__()
        self.num_baseterm_labels = num_baseterm_labels
        self.num_facet_labels = num_facet_labels

        # Load pre-trained model
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_baseterm_labels,
            return_dict=True,
        ).base_model

        # Base term classifier
        self.baseterm_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_baseterm_labels
        )

        # Facet classifier (multi-label)
        self.facet_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_facet_labels
        )

    def forward(
        self, input_ids, attention_mask, baseterm_label=None, facet_labels=None
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]

        # Base term prediction
        baseterm_logits = self.baseterm_classifier(pooled_output)

        # Facet prediction
        facet_logits = self.facet_classifier(pooled_output)

        loss = None
        if baseterm_label is not None and facet_labels is not None:
            # Base term loss (cross-entropy)
            baseterm_loss_fct = torch.nn.CrossEntropyLoss()
            baseterm_loss = baseterm_loss_fct(baseterm_logits, baseterm_label)

            # Facet loss (binary cross-entropy with logits)
            facet_loss_fct = torch.nn.BCEWithLogitsLoss()
            facet_loss = facet_loss_fct(facet_logits, facet_labels)

            # Total loss (weighted sum)
            loss = baseterm_loss + facet_loss

        return {
            "loss": loss,
            "baseterm_logits": baseterm_logits,
            "facet_logits": facet_logits,
        }


# Function to evaluate on random examples
def evaluate_random_examples(model, val_dataset, num_examples=5):
    """
    Run inference on random examples and print detailed results
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING ON {num_examples} RANDOM EXAMPLES")
    print("=" * 80)

    # Get model mappings
    id_to_baseterm = {idx: code for code, idx in val_dataset.baseterm_to_id.items()}
    id_to_facet = {idx: code for code, idx in val_dataset.facet_to_id.items()}

    # Ensure model is in evaluation mode
    model.eval()

    # Select random indices
    random_indices = random.sample(range(len(val_dataset)), num_examples)

    for i, idx in enumerate(random_indices):
        # Get the example
        example = val_dataset[idx]

        # Get the original text by decoding tokenized input
        input_ids = example["input_ids"].unsqueeze(0)
        attention_mask = example["attention_mask"].unsqueeze(0)
        original_text = val_dataset.texts[idx]

        # True labels
        true_baseterm_id = example["baseterm_label"]
        # Fix: Check if it's already an int or a tensor
        if isinstance(true_baseterm_id, torch.Tensor):
            true_baseterm_id = true_baseterm_id.item()
        true_baseterm = id_to_baseterm[true_baseterm_id]

        true_facets = []
        for facet_idx, is_present in enumerate(example["facet_labels"]):
            if is_present == 1:
                true_facets.append(id_to_facet[facet_idx])

        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get predictions and probabilities
        baseterm_logits = outputs["baseterm_logits"]
        facet_logits = outputs["facet_logits"]

        # Base term prediction with probabilities
        baseterm_probs = torch.softmax(baseterm_logits, dim=1)[0]
        predicted_baseterm_id = torch.argmax(baseterm_probs).item()
        predicted_baseterm = id_to_baseterm[predicted_baseterm_id]
        predicted_baseterm_prob = baseterm_probs[predicted_baseterm_id].item()

        # Top 3 base term predictions
        top_baseterm_indices = torch.topk(baseterm_probs, k=3)[1].tolist()
        top_baseterms = [
            (id_to_baseterm[idx], baseterm_probs[idx].item())
            for idx in top_baseterm_indices
        ]

        # Facet predictions with probabilities
        facet_probs = torch.sigmoid(facet_logits)[0]
        predicted_facet_indices = torch.where(facet_probs > 0.5)[0].tolist()
        predicted_facets = []

        for facet_idx in predicted_facet_indices:
            facet_code = id_to_facet[facet_idx]
            facet_prob = facet_probs[facet_idx].item()
            predicted_facets.append((facet_code, facet_prob))

        # Sort facets by probability
        predicted_facets.sort(key=lambda x: x[1], reverse=True)

        # Construct the full food code
        if predicted_facets:
            predicted_food_code = f"{predicted_baseterm}#{'$'.join([code for code, _ in predicted_facets])}"
        else:
            predicted_food_code = predicted_baseterm

        # Construct the true food code
        if true_facets:
            true_food_code = f"{true_baseterm}#{'$'.join(true_facets)}"
        else:
            true_food_code = true_baseterm

        # Print results
        print(f"\nExample {i+1}:")
        print(f"Text: {original_text}")
        print("\nGROUND TRUTH:")
        print(f"  Base Term: {true_baseterm}")
        print(f"  Facets: {', '.join(true_facets) if true_facets else 'None'}")
        print(f"  Full Food Code: {true_food_code}")

        print("\nPREDICTIONS:")
        print(
            f"  Base Term: {predicted_baseterm} (Probability: {predicted_baseterm_prob:.4f})"
        )
        print("\n  Top 3 Base Terms:")
        for j, (term, prob) in enumerate(top_baseterms):
            print(f"    {j+1}. {term} (Probability: {prob:.4f})")

        print("\n  Facets:")
        if predicted_facets:
            for facet_code, facet_prob in predicted_facets:
                print(f"    {facet_code} (Probability: {facet_prob:.4f})")
        else:
            print("    None")

        print(f"\n  Full Food Code: {predicted_food_code}")
        print("-" * 80)


# Training function
def train():
    # Load and preprocess data
    X, baseterm_codes, facet_codes = load_and_preprocess_data(DATA_PATH)

    # Create label encoders
    baseterm_to_id, id_to_baseterm, facet_to_id, id_to_facet = create_label_encoders(
        baseterm_codes, facet_codes
    )

    # Save mappings for inference
    mapping_data = {
        "baseterm_to_id": baseterm_to_id,
        "id_to_baseterm": id_to_baseterm,
        "facet_to_id": facet_to_id,
        "id_to_facet": id_to_facet,
    }
    torch.save(mapping_data, os.path.join(OUTPUT_DIR, "label_mappings.pt"))

    # Split data
    print("Splitting data into train and validation sets...")
    X_train, X_val, bt_train, bt_val, fc_train, fc_val = train_test_split(
        X, baseterm_codes, facet_codes, test_size=0.01, random_state=42
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Create datasets
    print("Creating datasets...")
    train_dataset = FoodClassificationDataset(
        X_train, bt_train, fc_train, baseterm_to_id, facet_to_id, tokenizer, MAX_LENGTH
    )
    val_dataset = FoodClassificationDataset(
        X_val, bt_val, fc_val, baseterm_to_id, facet_to_id, tokenizer, MAX_LENGTH
    )

    # Create model
    print("Creating model...")
    model = FoodClassificationModel(
        MODEL_NAME,
        num_baseterm_labels=len(baseterm_to_id),
        num_facet_labels=len(facet_to_id),
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Custom data collator
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        baseterm_labels = torch.tensor([item["baseterm_label"] for item in batch])
        facet_labels = torch.stack([item["facet_labels"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "baseterm_label": baseterm_labels,
            "facet_labels": facet_labels,
        }

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * NUM_EPOCHS

    # Move model to device
    model.to(device)

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0

        # Add progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch in train_pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            baseterm_label = batch["baseterm_label"].to(device)
            facet_labels = batch["facet_labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                baseterm_label=baseterm_label,
                facet_labels=facet_labels,
            )

            loss = outputs["loss"]
            train_loss += loss.item()

            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        # Add progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]")
        with torch.no_grad():
            for batch in val_pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                baseterm_label = batch["baseterm_label"].to(device)
                facet_labels = batch["facet_labels"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    baseterm_label=baseterm_label,
                    facet_labels=facet_labels,
                )

                loss = outputs["loss"]
                val_loss += loss.item()

                # Update progress bar with current loss
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))

    # Evaluate on random examples
    evaluate_random_examples(model, train_dataset, num_examples=5)


# Inference function
def predict(text):
    # Load label mappings
    mapping_data = torch.load(os.path.join(OUTPUT_DIR, "label_mappings.pt"))
    baseterm_to_id = mapping_data["baseterm_to_id"]
    id_to_baseterm = mapping_data["id_to_baseterm"]
    facet_to_id = mapping_data["facet_to_id"]
    id_to_facet = mapping_data["id_to_facet"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    # Load model
    model = FoodClassificationModel(
        MODEL_NAME,
        num_baseterm_labels=len(baseterm_to_id),
        num_facet_labels=len(facet_to_id),
    )
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    model.to(device)
    model.eval()

    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get predictions
    baseterm_logits = outputs["baseterm_logits"]
    facet_logits = outputs["facet_logits"]

    # Get probabilities
    baseterm_probs = torch.softmax(baseterm_logits, dim=1)[0]
    facet_probs = torch.sigmoid(facet_logits)[0]

    # Base term prediction
    baseterm_pred = torch.argmax(baseterm_probs).item()
    baseterm_code = id_to_baseterm[baseterm_pred]
    baseterm_probability = baseterm_probs[baseterm_pred].item()

    # Top 3 base term predictions
    top_baseterm_indices = torch.topk(baseterm_probs, k=3)[1].tolist()
    top_baseterms = [
        (id_to_baseterm[idx], baseterm_probs[idx].item())
        for idx in top_baseterm_indices
    ]

    # Facet predictions (multi-label)
    facet_preds = (facet_probs > 0.5).int().cpu().numpy()
    facet_indices = np.where(facet_preds == 1)[0]

    # Get facet codes with probabilities
    facet_results = []
    for idx in facet_indices:
        facet_code = id_to_facet[idx]
        probability = facet_probs[idx].item()
        facet_results.append((facet_code, probability))

    # Sort by probability
    facet_results.sort(key=lambda x: x[1], reverse=True)

    # Construct the full food code
    if facet_results:
        food_code = f"{baseterm_code}#{'$'.join([code for code, _ in facet_results])}"
    else:
        food_code = baseterm_code

    # Print detailed results
    print(f"\nPrediction for: '{text}'")
    print(f"\nBase Term: {baseterm_code} (Probability: {baseterm_probability:.4f})")

    print("\nTop 3 Base Terms:")
    for i, (term, prob) in enumerate(top_baseterms):
        print(f"  {i+1}. {term} (Probability: {prob:.4f})")

    print("\nFacet Predictions:")
    if facet_results:
        for facet_code, probability in facet_results:
            print(f"  {facet_code} (Probability: {probability:.4f})")
    else:
        print("  None")

    print(f"\nFull Food Code: {food_code}")

    return food_code


if __name__ == "__main__":
    train()
