import pandas as pd
import random
import re
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm

# Download necessary NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def parse_facets(facet_string):
    """
    Parse a FACETS identifier string into its component parts.
    
    Args:
        facet_string (str): The FACETS identifier string
    
    Returns:
        tuple: (unique_id, list of facet code/base term code pairs)
    """
    # Split by the hashtag to separate unique id from facet codes
    parts = facet_string.split("#")
    
    if len(parts) != 2:
        return None, []
    
    unique_id = parts[0]
    
    # Split the codes by $ to get individual facet/baseterm pairs
    facet_parts = parts[1].split("$")
    
    # Parse each facet/baseterm pair
    facet_baseterm_pairs = []
    for part in facet_parts:
        # Split by period to separate facet code from base term code
        codes = part.split(".")
        if len(codes) == 2:
            facet_code, baseterm_code = codes
            facet_baseterm_pairs.append((facet_code, baseterm_code))
    
    return unique_id, facet_baseterm_pairs

def generate_concise_description(facets_df, base_terms_df, facet_string):
    """
    Generate a concise food description from a FACETS identifier string.
    
    Args:
        facets_df (DataFrame): DataFrame containing facet codes and descriptions
        base_terms_df (DataFrame): DataFrame containing base term codes and descriptions
        facet_string (str): The FACETS identifier string
    
    Returns:
        str: Generated concise food description
    """
    # Parse the FACETS identifier
    _, facet_baseterm_pairs = parse_facets(facet_string)
    
    if not facet_baseterm_pairs:
        return ""
    
    # Construct descriptions
    descriptions = []
    
    for facet_code, baseterm_code in facet_baseterm_pairs:
        # Look up the base term information
        baseterm_info = base_terms_df[base_terms_df["termCode"] == baseterm_code]
        if not baseterm_info.empty:
            baseterm_name = baseterm_info.iloc[0]["termExtendedName"]
            # Remove bracketed content
            baseterm_name = re.sub(r'\s+\([^)]*\)', '', baseterm_name)
            descriptions.append(baseterm_name)
    
    # Join the descriptions with a separator
    return ", ".join(descriptions)

def apply_text_variations(text):
    """
    Apply various text transformations to create variations.
    Focus on keeping descriptions concise and natural.
    
    Args:
        text (str): Original text
        
    Returns:
        list: List of text variations
    """
    variations = []
    
    # Original text
    variations.append(text)
    
    # Case variations - make sure we have a good mix
    variations.append(text.lower())
    variations.append(text.upper())
    variations.append(text.title())
    variations.append(text.capitalize())
    
    # First word capitalized, rest lowercase
    words = text.split()
    if len(words) > 1:
        variations.append(words[0].capitalize() + ' ' + ' '.join(words[1:]).lower())
    
    # Remove all punctuation variation for a very compact representation
    no_punct = re.sub(r'[^\w\s]', ' ', text)
    variations.append(no_punct.strip())
    variations.append(' '.join(no_punct.split()))  # Also normalize whitespace
    
    # Join with different separators
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        variations.append(" and ".join(parts))
        variations.append(" with ".join(parts))
        
        # Reorder parts (if more than one)
        if len(parts) > 1:
            random.shuffle(parts)
            variations.append(", ".join(parts))
    
    return list(set(variations))  # Remove duplicates

def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and len(synonym) > 3:  # Only meaningful synonyms
                synonyms.add(synonym)
    return list(synonyms)

def replace_with_synonyms(text, probability=0.3, max_replacements=2):
    """Replace some words with their synonyms, keeping descriptions concise."""
    words = text.split()
    num_replacements = min(max_replacements, int(len(words) * probability))
    
    if num_replacements == 0 or len(words) < 3:
        return text
        
    indices_to_replace = random.sample(range(len(words)), num_replacements)
    
    for idx in indices_to_replace:
        word = words[idx]
        # Only replace words longer than 3 chars and not part of a code
        if len(word) > 3 and not any(c.isdigit() for c in word):
            synonyms = get_synonyms(word.lower())
            if synonyms:
                # Try to find a shorter synonym if possible
                short_synonyms = [s for s in synonyms if len(s) <= len(word)]
                if short_synonyms:
                    words[idx] = random.choice(short_synonyms)
                else:
                    words[idx] = random.choice(synonyms)
    
    return ' '.join(words)

def create_concise_natural_variants(text):
    """Create natural language variants, focusing on brevity and natural language."""
    variants = []
    
    # Split by comma
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        
        if len(parts) > 1:
            # Natural language templates
            main_ingredient = parts[0]
            attributes = parts[1:]
            
            # Very concise templates
            variants.append(f"{main_ingredient}")
            
            if len(attributes) == 1:
                variants.append(f"{attributes[0]} {main_ingredient}")
            else:
                variants.append(f"{main_ingredient} that is {' and '.join(attributes)}")
                
            # Just the main ingredient sometimes
            variants.append(main_ingredient)
            
            # Reordered version - make a different attribute the focus
            if len(parts) > 2:
                new_main = parts[1]
                new_attrs = [parts[0]] + parts[2:]
                variants.append(f"{new_main} with {', '.join(new_attrs)}")
    
    return variants

def generate_augmented_dataset(input_file, output_file, samples_per_item=5):
    """
    Generate an augmented dataset from the original food dataset,
    with a focus on concise descriptions.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
        samples_per_item (int): Number of augmented samples to generate per original item
    """
    print(f"Loading data from {input_file}...")
    food_df = pd.read_csv(input_file)
    
    # Load facets and base terms data
    facets_df = pd.read_csv("data/facets.csv")
    base_terms_df = pd.read_csv("data/facet_expressions.csv")
    
    # Prepare output data
    augmented_data = []
    
    # Keep track of first example to preserve its bracketed content
    first_example = True
    
    print(f"Generating {samples_per_item} variations for each of the {len(food_df)} food items...")
    for idx, row in tqdm(food_df.iterrows(), total=len(food_df)):
        facet_string = row["FACETS"]
        
        # Skip if FACETS is missing or empty
        if pd.isna(facet_string) or facet_string == "":
            continue
            
        # Generate concise description
        concise_desc = generate_concise_description(facets_df, base_terms_df, facet_string)
        
        if not concise_desc:
            continue
        
        # For the first example, we want to keep the bracketed content
        if first_example:
            # Parse to get the original terms with brackets
            _, facet_baseterm_pairs = parse_facets(facet_string)
            if facet_baseterm_pairs:
                original_terms = []
                for facet_code, baseterm_code in facet_baseterm_pairs:
                    baseterm_info = base_terms_df[base_terms_df["termCode"] == baseterm_code]
                    if not baseterm_info.empty:
                        original_terms.append(baseterm_info.iloc[0]["termExtendedName"])
                
                if original_terms:
                    concise_desc = ", ".join(original_terms)
            first_example = False
            
        # Generate variations of the descriptions
        concise_variations = apply_text_variations(concise_desc)
        
        # Create natural language variants
        natural_variants = create_concise_natural_variants(concise_desc)
        
        # Create synonym variations for a subset
        synonym_variations = []
        for desc in concise_variations[:2] + (natural_variants[:2] if natural_variants else []):
            synonym_variations.append(replace_with_synonyms(desc))
        
        # Combine all variations
        all_variations = concise_variations + natural_variants + synonym_variations
        
        # Remove any variations that are too long
        filtered_variations = [v for v in all_variations if len(v) < 150]
        if not filtered_variations:
            filtered_variations = [concise_desc]  # fallback to the original description
        
        # Randomly sample from all variations (or use all if fewer than requested)
        num_samples = min(samples_per_item, len(filtered_variations))
        selected_variations = random.sample(filtered_variations, num_samples)
        
        # Add the augmented data
        for variation in selected_variations:
            augmented_data.append({
                "ENFOODNAME": variation,
                "ENFOODNAME_SIMPLE": concise_desc,  # original concise description
                "FACETS": facet_string,
                "BASETERM_NAME": row["BASETERM_NAME"],
                "SCIENTIFIC_NAME": row.get("SCIENTIFIC_NAME", ""),
                "COMMON_NAME": row.get("COMMON_NAME", "")
            })
    
    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    
    # Save the augmented dataset
    augmented_df.to_csv(output_file, index=False)
    print(f"Augmented dataset with {len(augmented_df)} samples created at {output_file}")

def main():
    """Main function to run the data augmentation process."""
    input_file = "data/food2ex_curated_full.csv"
    output_file = "data/food2ex_augmented.csv"
    samples_per_item = 5  # Number of augmented samples per original item
    
    generate_augmented_dataset(input_file, output_file, samples_per_item)
    
    # Also create a smaller sample file for testing
    sample_output_file = "data/food2ex_augmented_sample.csv"
    augmented_df = pd.read_csv(output_file)
    sample_df = augmented_df.sample(n=min(1000, len(augmented_df)), random_state=42)
    sample_df.to_csv(sample_output_file, index=False)
    print(f"Sample augmented dataset with {len(sample_df)} samples created at {sample_output_file}")

if __name__ == "__main__":
    main()