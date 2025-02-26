import pandas as pd


def extract_facet_scopes(excel_file_path, sheet_name="attribute"):
    """
    Extract facet scopes from the attribute sheet where attributeType is "catalogue".

    Args:
        excel_file_path (str): Path to the Excel file
        sheet_name (str): Name of the worksheet to extract data from (default: "attribute")
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # Check if required columns exist
        required_columns = ["code", "name", "label", "scopeNote", "attributeType"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(
                f"Error: Missing columns in attribute sheet: {', '.join(missing_columns)}"
            )
            print(f"Available columns: {', '.join(df.columns)}")
            return

        # Filter for rows where attributeType is "catalogue"
        facets_df = df[df["attributeType"] == "catalogue"].copy()

        if len(facets_df) == 0:
            print("No facet scopes found with attributeType = 'catalogue'")
            return

        # Select only the required columns
        facets_df = facets_df[["code", "name", "label", "scopeNote"]]

        # Save to CSV
        output_file = "data/facets.csv"
        facets_df.to_csv(output_file, index=False)
        print(
            f"Facet scopes dataset with {len(facets_df)} rows created at {output_file}"
        )

    except FileNotFoundError:
        print(f"Error: File not found: {excel_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


def extract_data_from_excel(excel_file_path, sheet_name="term", sample_size=1000):
    """
    Extract data from an Excel file, rename columns, and save to CSV.
    Also creates a sample CSV with randomly selected rows.
    Additionally creates a separate CSV for base terms where termCode equals allFacets.

    Args:
        excel_file_path (str): Path to the Excel file
        sheet_name (str): Name of the worksheet to extract data from
        sample_size (int): Number of random rows to include in the sample file
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # Check if required columns exist
        required_columns = [
            "termScopeNote",
            "allFacets",
            "termExtendedName",
            "scientificNames",
            "commonNames",
            "status",
            "termCode",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing columns in Excel file: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(df.columns)}")
            return

        # Filter for rows where status is "APPROVED"
        df = df[df["status"] == "APPROVED"]
        print(f"Filtered to {len(df)} rows where status is 'APPROVED'")

        # Create a separate dataframe for base terms where termCode equals allFacets
        base_terms_df = df[df["termCode"] == df["allFacets"]].copy()
        base_terms_df = base_terms_df[["termCode", "termExtendedName"]]

        # Save base terms to CSV
        base_terms_output_file = "data/base_terms.csv"
        base_terms_df.to_csv(base_terms_output_file, index=False)
        print(
            f"Base terms dataset with {len(base_terms_df)} rows created at {base_terms_output_file}"
        )

        # Extract only the required columns (excluding status which was only used for filtering)
        extracted_data = df[
            [
                "termScopeNote",
                "allFacets",
                "termExtendedName",
                "scientificNames",
                "commonNames",
            ]
        ].copy()

        # Rename the columns
        column_mapping = {
            "termScopeNote": "ENFOODNAME",
            "allFacets": "FACETS",
            "termExtendedName": "BASETERM_NAME",
            "scientificNames": "SCIENTIFIC_NAME",
            "commonNames": "COMMON_NAME",
        }

        extracted_data = extracted_data.rename(columns=column_mapping)

        # Clean and validate data
        # 1. Convert non-string values in text columns to strings
        for col in [
            "ENFOODNAME",
            "FACETS",
            "BASETERM_NAME",
            "SCIENTIFIC_NAME",
            "COMMON_NAME",
        ]:
            # Replace NaN values with empty strings
            extracted_data[col] = extracted_data[col].fillna("")
            # Convert all values to strings
            extracted_data[col] = extracted_data[col].astype(str)

        # 2. Remove rows with empty values in any column
        initial_count = len(extracted_data)

        # Remove rows where any of the required columns has an empty value
        # Note: We allow SCIENTIFIC_NAMES and COMMON_NAMES to be empty as they might not be available for all entries
        extracted_data = extracted_data[
            (extracted_data["ENFOODNAME"].str.strip() != "")
            & (extracted_data["FACETS"].str.strip() != "")
            & (extracted_data["BASETERM_NAME"].str.strip() != "")
        ]

        if initial_count > len(extracted_data):
            print(
                f"Removed {initial_count - len(extracted_data)} rows with empty values in required columns"
            )

        # 3. Print data quality report
        print("\nData Quality Report:")
        print(f"Total rows: {len(extracted_data)}")
        for col in [
            "ENFOODNAME",
            "FACETS",
            "BASETERM_NAME",
            "SCIENTIFIC_NAME",
            "COMMON_NAME",
        ]:
            empty_count = (extracted_data[col].str.strip() == "").sum()
            print(
                f"- {col}: {empty_count} empty values ({empty_count/len(extracted_data)*100:.2f}%)"
            )

        output_file = "data/food2ex_dataset_extracted.csv"

        # Save to CSV
        extracted_data.to_csv(output_file, index=False)
        print(f"Data successfully extracted to {output_file}")

        # Create a sample dataset with random rows
        if len(extracted_data) > sample_size:
            sample_data = extracted_data.sample(n=sample_size, random_state=42)
        else:
            sample_data = extracted_data.copy()
            print(
                f"Warning: Dataset has fewer than {sample_size} rows. Using all available rows."
            )

        sample_output_file = "data/food2ex_dataset_sample.csv"
        sample_data.to_csv(sample_output_file, index=False)
        print(
            f"Sample dataset with {len(sample_data)} rows created at {sample_output_file}"
        )

    except FileNotFoundError:
        print(f"Error: File not found: {excel_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # You can modify this path to your Excel file
    excel_file = "data/Export_Catalogue_Browser_MTX_16.0.xlsx"

    # Extract main data from the "term" sheet
    extract_data_from_excel(excel_file)

    # Extract facet scopes from the "attribute" sheet
    extract_facet_scopes(excel_file)
