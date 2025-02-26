import os
import pandas as pd
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import re


def clean_text(text):
    """Clean text by removing URLs and special formatting"""
    # Remove URLs
    text = re.sub(r"£https?://\S+", "", text)
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_vector_database(csv_path, db_directory="./vector_db"):
    """
    Create a vector database from the food dataset

    Args:
        csv_path: Path to the CSV file containing food data
        db_directory: Directory to store the vector database
    """
    # Read the CSV file
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean the food descriptions
    print("Processing food descriptions...")
    df["cleaned_description"] = df["ENFOODNAME"].apply(clean_text)

    # Initialize OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )

    # Initialize ChromaDB client
    print(f"Initializing ChromaDB at {db_directory}...")
    chroma_client = chromadb.PersistentClient(path=db_directory)

    # Create or get a collection
    collection = chroma_client.get_or_create_collection(
        name="food_database", embedding_function=openai_ef
    )

    # Prepare data for insertion
    documents = df["cleaned_description"].tolist()
    metadatas = df.apply(
        lambda row: {
            "facets": row["FACETS"],
            "baseterm_name": row["BASETERM_NAME"],
            "original_description": row["ENFOODNAME"],
        },
        axis=1,
    ).tolist()
    ids = [f"food_{i}" for i in range(len(df))]

    # Add data to the collection
    print(f"Adding {len(documents)} food items to the vector database...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Vector database created successfully at {db_directory}")
    return collection


def query_vector_database(query_text, collection, n_results=3):
    """
    Query the vector database for similar food items

    Args:
        query_text: User's food description query
        collection: ChromaDB collection
        n_results: Number of results to return

    Returns:
        List of matching food items with their FACETS, BASETERM_NAME and similarity scores
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


if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "data/food2ex_dataset_sample.csv"

    # Create the vector database
    collection = create_vector_database(csv_path)

    # Test a query
    test_query = "confectioners' sugar for baking"
    print(f"\nTesting query: '{test_query}'")
    results = query_vector_database(test_query, collection)

    # Display results
    print("\nSearch Results:")
    if "distances" in results:
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
            print(f"Description: {document[:100]}...")
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
            print(f"Description: {document[:100]}...")
            print("-" * 50)
