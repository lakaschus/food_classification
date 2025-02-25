import pandas as pd


def extract_data_from_excel(excel_file_path, sheet_name="term"):
    """
    Extract termScopeNote and allFacets from an Excel file and save to CSV.

    Args:
        excel_file_path (str): Path to the Excel file
        sheet_name (str): Name of the worksheet to extract data from
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # Check if required columns exist
        required_columns = ["termScopeNote", "allFacets"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing columns in Excel file: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(df.columns)}")
            return

        # Extract only the required columns
        extracted_data = df[["termScopeNote", "allFacets"]]

        output_file = f"data/food2ex_dataset_extracted.csv"

        # Save to CSV
        extracted_data.to_csv(output_file, index=False)
        print(f"Data successfully extracted to {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found: {excel_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # You can modify this path to your Excel file
    excel_file = "data/Export_Catalogue_Browser_MTX_16.0.xlsx"

    extract_data_from_excel(excel_file)
