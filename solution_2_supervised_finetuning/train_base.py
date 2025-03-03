import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # ~66M parameters
MAX_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
DATA_PATH = "./data/base_terms.csv"  # Using base_terms.csv
OUTPUT_DIR = "./model_output_base_terms"
EVAL_STEPS = 500
SAVE_STEPS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load and preprocess the data
def load_and_preprocess_data(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required_columns = ["termCode", "termExtendedName"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Create a list of original samples
    X_original = df["termExtendedName"].tolist()
    y_original = df["termCode"].tolist()
    
    # Create augmented data with lowercase and uppercase variations
    X_augmented = []
    y_augmented = []
    
    # Add progress bar for data augmentation
    print("Creating augmented data...")
    for text, code in tqdm(zip(X_original, y_original), total=len(X_original), desc="Augmenting data"):
        # Skip if text is not a string or is empty
        if not isinstance(text, str) or not text.strip():
            continue
            
        # Original sample
        X_augmented.append(text)
        y_augmented.append(code)
        
        # Lowercase variation
        X_augmented.append(text.lower())
        y_augmented.append(code)
        
        # Uppercase variation
        X_augmented.append(text.upper())
        y_augmented.append(code)
        
        # Title case variation
        X_augmented.append(text.title())
        y_augmented.append(code)
    
    print(f"Loaded and augmented data: {len(X_augmented)} samples (from {len(X_original)} originals)")
    return X_augmented, y_augmented


# Create label encoders
def create_label_encoders(codes):
    unique_codes = sorted(list(set(codes)))  # Sort for reproducibility
    code_to_id = {code: idx for idx, code in enumerate(unique_codes)}
    id_to_code = {idx: code for code, idx in code_to_id.items()}

    print(f"Number of unique codes: {len(unique_codes)}")
    return code_to_id, id_to_code


# Custom dataset class
class BaseTermDataset(Dataset):
    def __init__(self, texts, codes, code_to_id, tokenizer, max_length=128):
        self.texts = texts
        self.codes = codes
        self.code_to_id = code_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        code = self.codes[idx]

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

        # Create label
        label = self.code_to_id[code]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


# Function to evaluate on random examples
def evaluate_random_examples(model, tokenizer, val_dataset, id_to_code, num_examples=5):
    """
    Run inference on random examples and print detailed results
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING ON {num_examples} RANDOM EXAMPLES")
    print("=" * 80)

    # Ensure model is in evaluation mode
    model.eval()

    # Select random indices
    random_indices = random.sample(range(len(val_dataset)), num_examples)

    for i, idx in enumerate(random_indices):
        # Get the example
        example = val_dataset[idx]
        original_text = val_dataset.texts[idx]
        true_code = val_dataset.codes[idx]

        # Move to device
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get predictions and probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        
        # Get top prediction
        predicted_id = torch.argmax(probs).item()
        predicted_code = id_to_code[predicted_id]
        predicted_prob = probs[predicted_id].item()

        # Top 3 predictions
        top_indices = torch.topk(probs, k=min(3, len(id_to_code)))[1].tolist()
        top_predictions = [(id_to_code[idx], probs[idx].item()) for idx in top_indices]

        # Print results
        print(f"\nExample {i+1}:")
        print(f"Text: {original_text}")
        print("\nGROUND TRUTH:")
        print(f"  Code: {true_code}")

        print("\nPREDICTIONS:")
        print(f"  Code: {predicted_code} (Probability: {predicted_prob:.4f})")
        
        print("\n  Top 3 Predictions:")
        for j, (code, prob) in enumerate(top_predictions):
            print(f"    {j+1}. {code} (Probability: {prob:.4f})")
        
        print("-" * 80)


# Training function
def train():
    # Load and preprocess data
    X, codes = load_and_preprocess_data(DATA_PATH)

    # Create label encoders
    code_to_id, id_to_code = create_label_encoders(codes)

    # Save mappings for inference
    mapping_data = {
        "code_to_id": code_to_id,
        "id_to_code": id_to_code,
    }
    torch.save(mapping_data, os.path.join(OUTPUT_DIR, "label_mappings.pt"))

    # Split data
    print("Splitting data into train and validation sets...")
    X_train, X_val, codes_train, codes_val = train_test_split(
        X, codes, test_size=0.2, random_state=42
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Create datasets
    print("Creating datasets...")
    train_dataset = BaseTermDataset(
        X_train, codes_train, code_to_id, tokenizer, MAX_LENGTH
    )
    val_dataset = BaseTermDataset(
        X_val, codes_val, code_to_id, tokenizer, MAX_LENGTH
    )

    # Create model
    print("Creating model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(code_to_id),
    )
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE
    )

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in train_pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()

            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]")
        
        with torch.no_grad():
            for batch in val_pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()

                # Update progress bar
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
            
            # Also save the full model for easier loading
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model_full"))

    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))

    # Evaluate on random examples
    evaluate_random_examples(model, tokenizer, val_dataset, id_to_code, num_examples=5)


# Inference function
def predict(text):
    # Load label mappings
    mapping_data = torch.load(os.path.join(OUTPUT_DIR, "label_mappings.pt"))
    code_to_id = mapping_data["code_to_id"]
    id_to_code = mapping_data["id_to_code"]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(OUTPUT_DIR, "best_model_full")
    )
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
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]

    # Get top prediction
    predicted_id = torch.argmax(probs).item()
    predicted_code = id_to_code[predicted_id]
    predicted_prob = probs[predicted_id].item()

    # Top 3 predictions
    top_indices = torch.topk(probs, k=min(3, len(id_to_code)))[1].tolist()
    top_predictions = [(id_to_code[idx], probs[idx].item()) for idx in top_indices]

    # Print detailed results
    print(f"\nPrediction for: '{text}'")
    print(f"\nCode: {predicted_code} (Probability: {predicted_prob:.4f})")

    print("\nTop 3 Predictions:")
    for i, (code, prob) in enumerate(top_predictions):
        print(f"  {i+1}. {code} (Probability: {prob:.4f})")

    return predicted_code


if __name__ == "__main__":
    train()