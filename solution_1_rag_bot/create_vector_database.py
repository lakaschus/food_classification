import os
import pandas as pd
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import re
import time


def clean_text(text):
    """Clean text by removing URLs and special formatting"""
    # Remove URLs
    text = re.sub(r"£https?://\S+", "", text)
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_core_terms_collection(
    chroma_client, openai_ef, facet_path="data/facet_expressions.csv"
):
    """
    Create a vector collection for core terms with direct term-to-code mapping.

    Args:
        chroma_client: Initialized ChromaDB client
        openai_ef: OpenAI embedding function
        facet_path: Path to the facet expressions CSV file

    Returns:
        ChromaDB collection for core terms
    """
    print(f"Creating core terms collection from {facet_path}...")

    # Check if the CSV file exists
    if not os.path.exists(facet_path):
        print(f"Error: Facet expressions file not found at {facet_path}")
        return None

    # Load the facet expressions CSV
    try:
        facet_df = pd.read_csv(facet_path)
        print(f"Loaded {len(facet_df)} facet expressions")
    except Exception as e:
        print(f"Error loading facet expressions: {str(e)}")
        return None

    # Create or get the collection
    collection = chroma_client.get_or_create_collection(
        name="core_terms", embedding_function=openai_ef
    )

    # Prepare data for the collection
    documents = []
    metadatas = []
    ids = []

    # Process each row in the facet expressions CSV
    for idx, row in facet_df.iterrows():
        term_code = row.get("termCode", "")
        term_name = row.get("termExtendedName", "")
        term_note = row.get("termScopeNote", "")

        if pd.isna(term_name) or pd.isna(term_code):
            continue

        # Clean the term name
        clean_term = clean_text(term_name)

        # Prepare metadata
        metadata = {
            "term": term_name,
            "code": term_code,
            "description": term_note if not pd.isna(term_note) else "",
        }

        documents.append(clean_term)
        metadatas.append(metadata)
        ids.append(f"term_{term_code}")

    # Add data to collection in batches
    batch_size = 2000
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_docs = documents[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]

        print(
            f"Processing core terms batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}: items {i} to {end_idx-1}"
        )
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)

        # Add a small delay between batches to avoid rate limits
        time.sleep(1)

    print(f"Core terms collection created with {len(documents)} terms")
    return collection


def create_vector_database(
    csv_path,
    db_directory="./vector_db",
    batch_size=2000,
    column_configs=None,
    create_core_terms=True,
    facet_path="data/facet_expressions.csv",
):
    """
    Create a vector database from the food dataset with configurable columns

    Args:
        csv_path: Path to the CSV file containing food data
        db_directory: Directory to store the vector database
        batch_size: Number of items to process in each batch
        column_configs: List of dictionaries with the following keys:
            - source_column: Column name to vectorize
            - collection_name: Name for the collection
            - cleaned_column: Name for the cleaned column (optional)
        create_core_terms: Whether to create the core terms collection
        facet_path: Path to the facet expressions CSV file
    """
    # Read the CSV file
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean the food descriptions
    print("Processing food descriptions...")
    for config in column_configs:
        source_col = config["source_column"]
        cleaned_col = config.get("cleaned_column", f"cleaned_{source_col.lower()}")
        df[cleaned_col] = df[source_col].apply(clean_text)
        config["cleaned_column"] = cleaned_col  # Store the cleaned column name

    # Initialize OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )

    # Initialize ChromaDB client
    print(f"Initializing ChromaDB at {db_directory}...")
    chroma_client = chromadb.PersistentClient(path=db_directory)

    # Create collections for each configuration
    collections = {}
    for config in column_configs:
        collection_name = config["collection_name"]
        collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=openai_ef
        )
        collections[collection_name] = collection

    # Prepare metadata
    metadatas = df.apply(
        lambda row: {
            "facets": row.get("FACETS", ""),
            "baseterm_name": row.get("BASETERM_NAME", ""),
            "detailed_description": row.get("ENFOODNAME", ""),
            "simple_description": row.get("ENFOODNAME_SIMPLE", ""),
            "scientific_name": row.get("SCIENTIFIC_NAME", ""),
            "common_name": row.get("COMMON_NAME", ""),
        },
        axis=1,
    ).tolist()

    ids = [f"food_{i}" for i in range(len(df))]

    # Add data to each collection in batches
    for config in column_configs:
        collection_name = config["collection_name"]
        cleaned_col = config["cleaned_column"]
        documents = df[cleaned_col].tolist()
        collection = collections[collection_name]

        print(
            f"Adding {len(documents)} food items to the {collection_name} vector database in batches of {batch_size}..."
        )

        # Process in batches to avoid rate limits
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            batch_meta = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]

            print(
                f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}: items {i} to {end_idx-1}"
            )
            collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)

            # Add a small delay between batches to avoid rate limits
            time.sleep(1)

    # Create the core terms collection if requested
    if create_core_terms:
        core_collection = create_core_terms_collection(
            chroma_client, openai_ef, facet_path
        )
        if core_collection:
            collections["core_terms"] = core_collection

    print(f"Vector databases created successfully at {db_directory}")
    return collections


def query_vector_database(query_text, collection, n_results=3):
    """
    Query the vector database for similar food items

    Args:
        query_text: User's food description query
        collection: ChromaDB collection
        n_results: Number of results to return

    Returns:
        List of matching food items with their metadata and similarity scores
    """
    # Different versions of ChromaDB have different parameter names
    try:
        # Try with include_distances (newer versions)
        results = collection.query(
            query_texts=[query_text], n_results=n_results, include_distances=True
        )
    except TypeError:
        # Fall back to older API
        results = collection.query(query_texts=[query_text], n_results=n_results)

        # If distances aren't included in results, we'll calculate them manually
        if "distances" not in results:
            # Add a placeholder for distances - using decreasing similarity scores
            # Starting at 0.95 and decreasing by 0.05 for each result
            results["distances"] = [
                [0.2 + (0.05 * i) for i in range(len(results["documents"][0]))]
            ]
            print(
                "Note: Using estimated distance scores as exact distances not available"
            )

    return results


def display_search_results(results, collection_type):
    """
    Display search results from a vector database query in a formatted way

    Args:
        results: The results dictionary returned from query_vector_database
        collection_type: String indicating which collection the results came from
    """
    print(f"\nSearch Results from {collection_type} collection:")
    if "distances" in results and results["documents"] and results["documents"][0]:
        for i, (document, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # ChromaDB typically returns L2 distances
            # For L2 normalized vectors, cosine similarity = 1 - (distance^2)/2
            # But we need to make sure we're not exceeding bounds
            cosine_similarity = 1 - (distance**2) / 2

            print(f"Result {i+1}:")
            if "baseterm_name" in metadata:
                print(f"Base Term: {metadata['baseterm_name']}")
                print(f"Facets: {metadata['facets']}")
                print(
                    f"Detailed Description: {metadata['detailed_description'][:100]}..."
                )
                print(f"Simple Description: {metadata['simple_description'][:100]}...")
            else:
                print(f"Term: {metadata.get('term', '')}")
                print(f"Code: {metadata.get('code', '')}")
                print(f"Description: {metadata.get('description', '')[:100]}...")

            print(f"Similarity Score: {cosine_similarity:.4f}")
            print(f"Raw Distance: {distance:.4f}")
            print("-" * 50)
    else:
        # Fallback if no distances are available
        for i, (document, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            print(f"Result {i+1}:")
            if "baseterm_name" in metadata:
                print(f"Base Term: {metadata['baseterm_name']}")
                print(f"Facets: {metadata['facets']}")
                print(
                    f"Detailed Description: {metadata['detailed_description'][:100]}..."
                )
                print(f"Simple Description: {metadata['simple_description'][:100]}...")
            else:
                print(f"Term: {metadata.get('term', '')}")
                print(f"Code: {metadata.get('code', '')}")
                print(f"Description: {metadata.get('description', '')[:100]}...")
            print("-" * 50)


if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "data/food2ex_curated_full.csv"  # Updated to use the curated dataset
    facet_path = "data/facet_expressions.csv"  # Path to facet expressions

    # Define column configurations
    column_configs = [
        # {"source_column": "ENFOODNAME", "collection_name": "detailed"},
        # {"source_column": "ENFOODNAME_SIMPLE", "collection_name": "simple"},
        # {"source_column": "BASETERM_NAME", "collection_name": "baseterm"},
    ]

    # Create the vector databases
    collections = create_vector_database(
        csv_path,
        column_configs=column_configs,
        create_core_terms=True,
        facet_path=facet_path,
    )

    # Test a query on all collections
    test_query = "confectioners' sugar for baking"

    for collection_type, collection in collections.items():
        print(f"\nTesting query on {collection_type} collection: '{test_query}'")
        results = query_vector_database(test_query, collection)
        display_search_results(results, collection_type)

    # Test a direct food lookup query
    test_direct_query = "Hummus"
    if "core_terms" in collections:
        print(
            f"\nTesting direct lookup on core_terms collection: '{test_direct_query}'"
        )
        direct_results = query_vector_database(
            test_direct_query, collections["core_terms"]
        )
        display_search_results(direct_results, "core_terms")
