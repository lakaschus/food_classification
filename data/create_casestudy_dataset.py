import pandas as pd
import os


def process_foodex_dataset(input_file, output_file):
    """
    Process the FoodEx2 case study dataset:
    1. Read the Excel file
    2. Remove duplicates based on BASETERM_NAME, keeping entries with shortest FACETS
    3. Save as CSV
    """
    print(f"Reading Excel file: {input_file}")

    # Read the Excel file
    df = pd.read_excel(input_file)

    print(f"Original dataset shape: {df.shape}")

    # Sort by FACETS length to ensure we keep the shortest when removing duplicates
    df["FACETS_LENGTH"] = df["FACETS"].astype(str).apply(len)
    df = df.sort_values("FACETS_LENGTH")

    # Remove duplicates based on BASETERM_NAME, keeping the first occurrence (shortest FACETS)
    df_no_duplicates = df.drop_duplicates(subset=["BASETERM_NAME"], keep="first")

    # Remove the temporary column
    df_no_duplicates = df_no_duplicates.drop(columns=["FACETS_LENGTH"])

    print(f"Dataset after removing duplicates: {df_no_duplicates.shape}")
    print(f"Removed {df.shape[0] - df_no_duplicates.shape[0]} duplicate rows")

    # Save to CSV
    df_no_duplicates.to_csv(output_file, index=False)
    print(f"Saved processed dataset to: {output_file}")


if __name__ == "__main__":
    # Define input and output file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    input_file = os.path.join(parent_dir, "data", "FoodEx2-CaseStudy2-Dataset_V1.xlsx")
    output_file = os.path.join(
        parent_dir, "data", "foodex2_training_dataset_cleaned.csv"
    )

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please make sure the Excel file is in the correct location.")
        exit(1)

    # Process the dataset
    process_foodex_dataset(input_file, output_file)
