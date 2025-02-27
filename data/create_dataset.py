import pandas as pd


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


def construct_detailed_food_description(facets_df, base_terms_df, food_row):
    """
    Construct the detailed food description from FACETS identifiers.

    Args:
        facets_df (DataFrame): DataFrame containing facet codes and descriptions
        base_terms_df (DataFrame): DataFrame containing base term codes and descriptions
        food_row (Series): A row from the food dataset

    Returns:
        str: ENFOODNAME (detailed description)
    """
    facet_string = food_row["FACETS"]

    # If FACETS is missing or empty, return empty string
    if pd.isna(facet_string) or facet_string == "":
        return ""

    # Parse the FACETS identifier
    _, facet_baseterm_pairs = parse_facets(facet_string)

    if not facet_baseterm_pairs:
        return ""

    # Construct the detailed description
    detailed_descriptions = []

    for facet_code, baseterm_code in facet_baseterm_pairs:
        # Look up the facet information
        facet_info = facets_df[facets_df["code"] == facet_code]
        if not facet_info.empty:
            facet_name = facet_info.iloc[0]["name"]
            facet_scopenote = facet_info.iloc[0]["scopeNote"]

            # Look up the base term information
            baseterm_info = base_terms_df[base_terms_df["termCode"] == baseterm_code]
            if not baseterm_info.empty:
                baseterm_name = baseterm_info.iloc[0]["termExtendedName"]
                baseterm_scopenote = baseterm_info.iloc[0]["termScopeNote"]

                # Construct the detailed description in the updated format:
                # facet_name: baseterm_name (facet_scopenote) (baseterm_scopenote)
                detailed_description = f"{facet_name} ({facet_scopenote}): {baseterm_name} ({baseterm_scopenote})"
                detailed_descriptions.append(detailed_description)

    # Join the descriptions
    detailed_text = " | ".join(detailed_descriptions)

    return detailed_text


def construct_simple_food_description(facets_df, base_terms_df, food_row):
    """
    Construct the simple food description from FACETS identifiers.

    Args:
        facets_df (DataFrame): DataFrame containing facet codes and descriptions
        base_terms_df (DataFrame): DataFrame containing base term codes and descriptions
        food_row (Series): A row from the food dataset

    Returns:
        str: ENFOODNAME_SIMPLE
    """
    facet_string = food_row["FACETS"]

    # If FACETS is missing or empty, return empty string
    if pd.isna(facet_string) or facet_string == "":
        return ""

    # Parse the FACETS identifier
    _, facet_baseterm_pairs = parse_facets(facet_string)

    if not facet_baseterm_pairs:
        return ""

    # Construct the simple description
    simple_descriptions = []

    for facet_code, baseterm_code in facet_baseterm_pairs:
        # Look up the facet information
        facet_info = facets_df[facets_df["code"] == facet_code]
        if not facet_info.empty:
            facet_name = facet_info.iloc[0]["name"]

            # Look up the base term information
            baseterm_info = base_terms_df[base_terms_df["termCode"] == baseterm_code]
            if not baseterm_info.empty:
                baseterm_name = baseterm_info.iloc[0]["termExtendedName"]

                # Construct the simple description
                simple_description = f"{facet_name}: {baseterm_name}"
                simple_descriptions.append(simple_description)

    # Join the descriptions
    simple_text = " | ".join(simple_descriptions)

    return simple_text


def create_curated_dataset(sample_size=None):
    """
    Create the curated dataset by combining information from facet_expressions.csv,
    facets.csv, and food2ex_dataset_extracted.csv.

    Args:
        sample_size (int, optional): Number of rows to sample. If None, process the entire dataset.
    """
    print("Loading data files...")
    # Load the necessary CSV files
    base_terms_df = pd.read_csv("data/facet_expressions.csv")
    facets_df = pd.read_csv("data/facets.csv")

    # Determine input and output file names based on sample_size
    input_file = "data/food2ex_dataset_extracted.csv"
    if sample_size is not None:
        output_file = f"data/food2ex_curated_sample_{sample_size}.csv"
    else:
        output_file = "data/food2ex_curated_full.csv"

    try:
        # Load the dataset
        food_df = pd.read_csv(input_file)
        print(f"Loaded dataset with {len(food_df)} rows")

        # Apply sampling if sample_size is provided
        if sample_size is not None:
            food_df = food_df.sample(n=min(sample_size, len(food_df)), random_state=42)
            print(f"Sampled {len(food_df)} rows")

        # Store original ENFOODNAME for backup
        food_df["ORIGINAL_ENFOODNAME"] = food_df["ENFOODNAME"]

        print("Processing FACETS identifiers...")
        # Create empty columns for the new descriptions
        food_df["ENFOODNAME_SIMPLE"] = ""

        # Process each row in the food dataset
        for idx, row in food_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing row {idx}/{len(food_df)}...")

            # Construct the detailed and simple food descriptions
            enfoodname = construct_detailed_food_description(
                facets_df, base_terms_df, row
            )
            enfoodname_simple = construct_simple_food_description(
                facets_df, base_terms_df, row
            )

            # Update the DataFrame with the new descriptions
            # Only replace ENFOODNAME if we successfully constructed a new one
            if enfoodname:
                food_df.at[idx, "ENFOODNAME"] = enfoodname
            food_df.at[idx, "ENFOODNAME_SIMPLE"] = enfoodname_simple

        # Filter out rows where ENFOODNAME or ENFOODNAME_SIMPLE is empty
        food_df = food_df[
            food_df["ENFOODNAME"].notna()
            & food_df["ENFOODNAME"].ne("")
            & food_df["ENFOODNAME_SIMPLE"].notna()
            & food_df["ENFOODNAME_SIMPLE"].ne("")
        ]

        print(f"After filtering null values, dataset has {len(food_df)} rows")

        # Reorder columns as requested
        columns_order = [
            "ENFOODNAME",
            "ENFOODNAME_SIMPLE",
            "FACETS",
            "BASETERM_NAME",
            "SCIENTIFIC_NAME",
            "COMMON_NAME",
        ]
        food_df = food_df[columns_order]

        # Save the curated dataset
        food_df.to_csv(output_file, index=False)
        print(f"Curated dataset created at {output_file}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    create_curated_dataset(
        sample_size=None
    )  # Default to 1000 sample size, change or set to None for full dataset
