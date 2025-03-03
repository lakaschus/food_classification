import pandas as pd
import os
import time
import random
import json
from tqdm import tqdm
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Set your OpenAI API key
# openai.api_key = "your-api-key-here"  # Uncomment and add your API key or set as environment variable

# Configuration
INPUT_FILE = "data/food2ex_augmented.csv"  # File with existing augmented data
OUTPUT_FILE = "data/food2ex_augmented_with_llm.csv"  # New file with LLM-augmented data
CHECKPOINT_FILE = "data/llm_augmentation_checkpoint.jsonl"  # To save progress
VARIANTS_PER_ITEM = 3
BATCH_SIZE = 10  # Number of items to process before saving checkpoint

def setup_openai_client():
    """Set up and return OpenAI client."""
    # Get API key from environment variable if not set above
    api_key = os.environ.get("OPENAI_API_KEY", openai.api_key)
    if not api_key:
        raise ValueError("OpenAI API key is required. Set it in the script or as an environment variable.")
    
    return openai.OpenAI(api_key=api_key)

@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def generate_llm_variants(client, food_description, num_variants=3):
    """
    Generate realistic food description variants using GPT-4o mini.
    
    Args:
        client: OpenAI client
        food_description (str): Original food description
        num_variants (int): Number of variants to generate
    
    Returns:
        list: Generated food description variants
    """
    try:
        prompt = f"""As a food expert, please generate {num_variants} alternative ways that a person might describe this food in a natural, conversational way:

"{food_description}"

Generate descriptions that:
- Vary in length and style
- Use everyday language that people would actually use
- Are not too technical or scientific
- Focus on the most important food characteristics
- Are brief but descriptive

Format your response as a JSON array containing only the description strings.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic food descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        # Extract the variants from the response
        content = response.choices[0].message.content
        # Parse the JSON response
        try:
            variants_data = json.loads(content)
            # The response might be nested in different ways depending on how the model formats it
            if "descriptions" in variants_data:
                variants = variants_data["descriptions"]
            elif "variants" in variants_data:
                variants = variants_data["variants"]
            else:
                # Assume it's a direct array or the first key in the JSON
                variants = list(variants_data.values())[0] if isinstance(variants_data, dict) else variants_data
                
            # Ensure we have strings
            variants = [str(v).strip() for v in variants if v]
            
            # If we somehow got no valid variants, return a default
            if not variants:
                return [f"variant of {food_description}"]
                
            return variants[:num_variants]  # Ensure we don't exceed requested number
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            print(f"Response content: {content}")
            # Return a simple variation as fallback
            return [f"variant of {food_description}"]
            
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
        print(f"API error: {e}. Retrying...")
        raise  # Will be caught by the retry decorator
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [f"variant of {food_description}"]  # Return a simple variation as fallback

def load_checkpoint():
    """Load the checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            processed_items = set()
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_items.add(item['original_desc'])
                except json.JSONDecodeError:
                    continue
            return processed_items
    return set()

def save_checkpoint(original_desc, variants):
    """Append to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        checkpoint_data = {
            'original_desc': original_desc,
            'variants': variants,
            'timestamp': time.time()
        }
        f.write(json.dumps(checkpoint_data) + '\n')

def main():
    # Check if output file already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"Output file {OUTPUT_FILE} already exists. Will append new data without duplicates.")
        existing_df = pd.read_csv(OUTPUT_FILE)
        # Create a set of existing descriptions for faster lookup
        existing_descriptions = set(existing_df['ENFOODNAME'].tolist())
    else:
        existing_df = None
        existing_descriptions = set()
    
    # Load the input data
    print(f"Loading data from {INPUT_FILE}...")
    food_df = pd.read_csv(INPUT_FILE)
    
    # Load checkpoint to resume from last run
    processed_items = load_checkpoint()
    print(f"Found {len(processed_items)} already processed items in checkpoint.")
    
    # Set up OpenAI client
    client = setup_openai_client()
    
    # Prepare container for new data
    new_data = []
    batch_counter = 0
    
    # Choose a subset of rows to process
    # Group by FACETS to avoid generating variants for duplicates
    unique_foods = food_df.drop_duplicates(subset=['FACETS', 'BASETERM_NAME'])
    
    print(f"Generating {VARIANTS_PER_ITEM} LLM variants for each of {len(unique_foods)} unique food items...")
    
    for idx, row in tqdm(unique_foods.iterrows(), total=len(unique_foods)):
        food_description = row["ENFOODNAME"]
        
        # Skip if already processed in a previous run
        if food_description in processed_items:
            print(f"Skipping already processed: {food_description}")
            continue
        
        # Generate LLM variants
        try:
            variants = generate_llm_variants(client, food_description, VARIANTS_PER_ITEM)
            
            # Filter out variants that already exist
            variants = [v for v in variants if v not in existing_descriptions]
            
            # Add the new data
            for variant in variants:
                new_row = row.copy()
                new_row["ENFOODNAME"] = variant
                new_data.append(new_row)
            
            # Save checkpoint
            save_checkpoint(food_description, variants)
            
            # Increment batch counter
            batch_counter += 1
            
            # Save intermediate results periodically
            if batch_counter % BATCH_SIZE == 0:
                print(f"Saving intermediate results after processing {batch_counter} batches...")
                temp_df = pd.DataFrame(new_data)
                
                if existing_df is not None:
                    # Combine with existing data
                    combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
                    combined_df.to_csv(OUTPUT_FILE, index=False)
                    existing_df = combined_df
                    existing_descriptions.update(temp_df['ENFOODNAME'].tolist())
                else:
                    # Create new file
                    temp_df.to_csv(OUTPUT_FILE, index=False)
                    existing_df = temp_df
                    existing_descriptions.update(temp_df['ENFOODNAME'].tolist())
                
                # Clear the new_data list to free memory
                new_data = []
            
            # Add a small delay to avoid hammering the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing {food_description}: {e}")
            continue
    
    # Save any remaining new data
    if new_data:
        print(f"Saving final results with {len(new_data)} additional variants...")
        temp_df = pd.DataFrame(new_data)
        
        if existing_df is not None:
            # Combine with existing data
            combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
            combined_df.to_csv(OUTPUT_FILE, index=False)
        else:
            # Create new file
            temp_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"LLM augmentation complete! Output saved to {OUTPUT_FILE}")
    print(f"Checkpoint file saved to {CHECKPOINT_FILE}")

if __name__ == "__main__":
    main()