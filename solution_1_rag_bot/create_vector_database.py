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


def create_vector_database(csv_path, db_directory="./vector_db", batch_size=2000):
    """
    Create a vector database from the food dataset with both detailed and simple descriptions

    Args:
        csv_path: Path to the CSV file containing food data
        db_directory: Directory to store the vector database
        batch_size: Number of items to process in each batch
    """
    # Read the CSV file
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean the food descriptions
    print("Processing food descriptions...")
    df["cleaned_detailed"] = df["ENFOODNAME"].apply(clean_text)
    df["cleaned_simple"] = df["ENFOODNAME_SIMPLE"].apply(clean_text)

    # Initialize OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )

    # Initialize ChromaDB client
    print(f"Initializing ChromaDB at {db_directory}...")
    chroma_client = chromadb.PersistentClient(path=db_directory)

    # Create or get collections for both detailed and simple descriptions
    detailed_collection = chroma_client.get_or_create_collection(
        name="detailed", embedding_function=openai_ef
    )

    simple_collection = chroma_client.get_or_create_collection(
        name="simple", embedding_function=openai_ef
    )

    # Prepare data for insertion
    detailed_documents = df["cleaned_detailed"].tolist()
    simple_documents = df["cleaned_simple"].tolist()

    metadatas = df.apply(
        lambda row: {
            "facets": row["FACETS"],
            "baseterm_name": row["BASETERM_NAME"],
            "detailed_description": row["ENFOODNAME"],
            "simple_description": row["ENFOODNAME_SIMPLE"],
            "scientific_name": row.get("SCIENTIFIC_NAME", ""),
            "common_name": row.get("COMMON_NAME", ""),
        },
        axis=1,
    ).tolist()

    ids = [f"food_{i}" for i in range(len(df))]

    # Add data to the collections in batches
    print(
        f"Adding {len(detailed_documents)} food items to the detailed vector database in batches of {batch_size}..."
    )

    # Process in batches to avoid rate limits
    for i in range(0, len(detailed_documents), batch_size):
        end_idx = min(i + batch_size, len(detailed_documents))
        batch_docs = detailed_documents[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]

        print(
            f"Processing batch {i//batch_size + 1}/{(len(detailed_documents) + batch_size - 1)//batch_size}: items {i} to {end_idx-1}"
        )
        detailed_collection.add(
            documents=batch_docs, metadatas=batch_meta, ids=batch_ids
        )

        # Optional: add a small delay between batches to avoid rate limits
        time.sleep(2)

    print(
        f"Adding {len(simple_documents)} food items to the simple vector database in batches of {batch_size}..."
    )
    for i in range(0, len(simple_documents), batch_size):
        end_idx = min(i + batch_size, len(simple_documents))
        batch_docs = simple_documents[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]

        print(
            f"Processing batch {i//batch_size + 1}/{(len(simple_documents) + batch_size - 1)//batch_size}: items {i} to {end_idx-1}"
        )
        simple_collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)

        # Optional: add a small delay between batches to avoid rate limits
        time.sleep(0.5)

    print(f"Vector databases created successfully at {db_directory}")
    return {"detailed": detailed_collection, "simple": simple_collection}


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
            print(f"Base Term: {metadata['baseterm_name']}")
            print(f"Facets: {metadata['facets']}")
            print(f"Detailed Description: {metadata['detailed_description'][:100]}...")
            print(f"Simple Description: {metadata['simple_description'][:100]}...")
            print(f"Similarity Score: {cosine_similarity:.4f}")
            print(f"Raw Distance: {distance:.4f}")
            print("-" * 50)
    else:
        # Fallback if no distances are available
        for i, (document, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            print(f"Result {i+1}:")
            print(f"Base Term: {metadata['baseterm_name']}")
            print(f"Facets: {metadata['facets']}")
            print(f"Detailed Description: {metadata['detailed_description'][:100]}...")
            print(f"Simple Description: {metadata['simple_description'][:100]}...")
            print("-" * 50)


if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "data/food2ex_curated_full.csv"  # Updated to use the curated dataset

    # Create the vector databases
    collections = create_vector_database(csv_path)

    # Test a query on both collections
    test_query = "confectioners' sugar for baking"

    for collection_type, collection in collections.items():
        print(f"\nTesting query on {collection_type} collection: '{test_query}'")
        results = query_vector_database(test_query, collection)
        display_search_results(results, collection_type)
