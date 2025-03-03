import os
import argparse
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sys

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
OUTPUT_DIR = "./model_output"
DATA_PATH = "./data/food2ex_curated_full.csv"  # Path to dataset


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
        from transformers import AutoModelForSequenceClassification
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


def load_and_preprocess_data(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Extract features and labels
    X = df["BASETERM_NAME"].tolist()

    # Parse FACETS column to extract baseterm and facets
    baseterm_codes = []
    facet_codes = []

    for facets_str in df["FACETS"]:
        # Split into baseterm and facets
        if "#" in facets_str:
            baseterm, facets = facets_str.split("#", 1)
            baseterm_codes.append(baseterm)

            # Extract facet codes
            facet_list = []
            if facets:
                for facet in facets.split("$"):
                    if "." in facet:
                        facet_code = facet.split(".", 1)[0]
                        facet_exp = facet
                        facet_list.append(facet_exp)

            facet_codes.append(facet_list)
        else:
            baseterm_codes.append(facets_str)
            facet_codes.append([])

    print(f"Loaded {len(X)} samples")
    return X, baseterm_codes, facet_codes


def load_model_and_tokenizer():
    """Load the model, tokenizer, and label mappings"""
    # Check if model files exist
    if not os.path.exists(os.path.join(OUTPUT_DIR, "best_model.pt")) or \
       not os.path.exists(os.path.join(OUTPUT_DIR, "label_mappings.pt")):
        print("Error: Model files not found. Please train the model first.")
        sys.exit(1)
        
    print("Loading model and tokenizer...")
    
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
    
    return model, tokenizer, baseterm_to_id, id_to_baseterm, facet_to_id, id_to_facet


def predict_text(text, model, tokenizer, id_to_baseterm, id_to_facet):
    """Generate prediction for a user-entered food description"""
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
    facet_indices = torch.where(facet_probs > 0.5)[0].tolist()

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


def evaluate_random_examples(num_examples=5):
    """Run inference on random examples from the dataset"""
    # Load model and data
    model, tokenizer, baseterm_to_id, id_to_baseterm, facet_to_id, id_to_facet = load_model_and_tokenizer()
    X, baseterm_codes, facet_codes = load_and_preprocess_data(DATA_PATH)
    
    # Create dataset
    dataset = FoodClassificationDataset(
        X, baseterm_codes, facet_codes, baseterm_to_id, facet_to_id, tokenizer, MAX_LENGTH
    )
    
    print("\n" + "=" * 80)
    print(f"EVALUATING ON {num_examples} RANDOM EXAMPLES")
    print("=" * 80)

    # Select random indices
    random_indices = random.sample(range(len(dataset)), num_examples)

    for i, idx in enumerate(random_indices):
        # Get the example
        example = dataset[idx]

        # Get the original text
        original_text = dataset.texts[idx]

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

        # Get input tensors
        input_ids = example["input_ids"].unsqueeze(0)
        attention_mask = example["attention_mask"].unsqueeze(0)

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


def run_interactive_mode():
    """Run the interactive mode for user input"""
    model, tokenizer, baseterm_to_id, id_to_baseterm, facet_to_id, id_to_facet = load_model_and_tokenizer()
    
    print("\nFood Classification Interactive Mode")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Enter food description: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
            
        if not text:
            print("Please enter a valid food description")
            continue
            
        predict_text(text, model, tokenizer, id_to_baseterm, id_to_facet)
        print("\n" + "-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Food Classification Inference")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["random", "interactive"], 
        default="interactive",
        help="Inference mode: 'random' for random examples, 'interactive' for user input"
    )
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=5, 
        help="Number of random examples to evaluate"
    )
    args = parser.parse_args()
    
    if args.mode == "random":
        evaluate_random_examples(args.num_examples)
    else:
        run_interactive_mode()


if __name__ == "__main__":
    # You can call functions directly here:
    # For example, to run the interactive mode directly:
    run_interactive_mode()
    
    # Or, to run with random examples:
    # evaluate_random_examples(5)
    
    # Default behavior: process command-line arguments
    # main()