import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def load_vector_database(db_directory="./vector_db"):
    """
    Load an existing vector database

    Args:
        db_directory: Directory where the vector database is stored

    Returns:
        ChromaDB collection for querying
    """
    # Initialize OpenAI embedding function
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )

    # Initialize ChromaDB client
    print(f"Connecting to ChromaDB at {db_directory}...")
    chroma_client = chromadb.PersistentClient(path=db_directory)

    # Get the existing collection
    collection = chroma_client.get_collection(
        name="food_database", embedding_function=openai_ef
    )

    print(f"Successfully connected to vector database with {collection.count()} items")
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
    # Clean the query text if needed
    query_text = query_text.strip()

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


def format_results(results):
    """
    Format the query results for display

    Args:
        results: Results from query_vector_database

    Returns:
        Formatted string of results
    """
    output = "\nSearch Results:\n"

    if not results["documents"][0]:
        return "No matching results found."

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
            cosine_similarity = 1 - (distance**2) / 2

            output += f"Result {i+1}:\n"
            output += f"Base Term: {metadata['baseterm_name']}\n"
            output += f"Facets: {metadata['facets']}\n"
            output += f"Description: {metadata['original_description']}\n"
            output += f"Similarity Score: {cosine_similarity:.4f}\n"
            output += "-" * 50 + "\n"
    else:
        # Fallback if no distances are available
        for i, (document, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            output += f"Result {i+1}:\n"
            output += f"Base Term: {metadata['baseterm_name']}\n"
            output += f"Facets: {metadata['facets']}\n"
            output += f"Description: {metadata['original_description']}\n"
            output += "-" * 50 + "\n"

    return output


if __name__ == "__main__":
    # Load the existing vector database
    collection = load_vector_database()

    # Interactive query loop
    print("\nFood Database Query Tool")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter food description to search: ")

        if query.lower() in ["exit", "quit", "q"]:
            break

        if not query.strip():
            continue

        # Query the database
        results = query_vector_database(query, collection, n_results=10)

        # Display results
        print(format_results(results))
