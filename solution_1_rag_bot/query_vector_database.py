import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import json
from openai import OpenAI


def load_vector_database(db_directory="./vector_db", collection_type="detailed"):
    """
    Load an existing vector database

    Args:
        db_directory: Directory where the vector database is stored
        collection_type: Type of collection to load ("detailed" or "simple")

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

    # Get the existing collection based on collection_type
    collection_name = f"{collection_type}"
    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=openai_ef
    )

    print(
        f"Successfully connected to {collection_name} with {collection.count()} items"
    )
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


def format_results(results, collection_type):
    """
    Format the query results for display

    Args:
        results: Results from query_vector_database
        collection_type: Type of collection queried ("detailed" or "simple")

    Returns:
        Formatted string of results
    """
    output = f"\nSearch Results from {collection_type} collection:\n"

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

            # Display the appropriate description based on collection type
            if collection_type == "detailed":
                output += f"Detailed Description: {metadata['detailed_description']}\n"
            else:
                output += f"Simple Description: {metadata['simple_description']}\n"

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

            # Display the appropriate description based on collection type
            if collection_type == "detailed":
                output += f"Detailed Description: {metadata['detailed_description']}\n"
                output += f"Simple Description: {metadata['simple_description']}\n"
            else:
                output += f"Simple Description: {metadata['simple_description']}\n"
                output += f"Detailed Description: {metadata['detailed_description']}\n"

            output += "-" * 50 + "\n"

    return output


def load_all_collections(db_directory="./vector_db"):
    """
    Load both detailed and simple collections

    Args:
        db_directory: Directory where the vector database is stored

    Returns:
        Dictionary with both collections
    """
    collections = {
        "detailed": load_vector_database(db_directory, "detailed"),
        "simple": load_vector_database(db_directory, "simple"),
        "baseterm": load_vector_database(db_directory, "baseterm"),
    }
    return collections


def advanced_multi_collection_search(
    query_text, collections, n_results=10, model="gpt-4o"
):
    """
    Query multiple collections and use a language model to select the most relevant results.

    Args:
        query_text: User's food description query
        collections: Dictionary of ChromaDB collections (detailed, simple, baseterm)
        n_results: Number of results to retrieve from each collection
        model: OpenAI model to use for reranking

    Returns:
        Dictionary containing the original query, all retrieved results, and ranked results
    """
    all_results = {}
    merged_candidates = []

    # Query each collection
    for collection_name, collection in collections.items():
        results = query_vector_database(query_text, collection, n_results=n_results)
        all_results[collection_name] = results

        # Process results from this collection
        if results["documents"] and results["documents"][0]:
            for i, (document, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Calculate similarity score
                cosine_similarity = 1 - (distance**2) / 2

                # Create a candidate entry with all relevant information
                candidate = {
                    "collection": collection_name,
                    "baseterm_name": metadata.get("baseterm_name", ""),
                    "facets": metadata.get("facets", ""),
                    "detailed_description": metadata.get("detailed_description", ""),
                    "simple_description": metadata.get("simple_description", ""),
                    "scientific_name": metadata.get("scientific_name", ""),
                    "common_name": metadata.get("common_name", ""),
                    "similarity": cosine_similarity,
                    "original_rank": i + 1,
                }
                merged_candidates.append(candidate)

    # If we have results, use the LLM to rank them
    if merged_candidates:
        client = OpenAI()

        # Convert candidates to a readable format for the LLM
        candidates_text = ""
        for i, candidate in enumerate(merged_candidates):
            candidates_text += f"Candidate {i+1}:\n"
            candidates_text += f"- Base Term: {candidate['baseterm_name']}\n"
            candidates_text += f"- Facets: {candidate['facets']}\n"
            candidates_text += (
                f"- Detailed Description: {candidate['detailed_description']}\n"
            )
            candidates_text += (
                f"- Simple Description: {candidate['simple_description']}\n"
            )

            if candidate["scientific_name"]:
                candidates_text += (
                    f"- Scientific Name: {candidate['scientific_name']}\n"
                )
            if candidate["common_name"]:
                candidates_text += f"- Common Name: {candidate['common_name']}\n"

            candidates_text += f"- Vector Similarity: {candidate['similarity']:.4f}\n"
            candidates_text += f"- Collection: {candidate['collection']}\n"
            candidates_text += "-" * 50 + "\n"

        # Create prompt for the LLM
        prompt = f"""
You are a food code classification expert. Given a user's food description query, rank the most likely food codes that match the user's description.

USER QUERY: "{query_text}"

Below are {len(merged_candidates)} potential food matches retrieved from a vector database. 
Each candidate includes:
- Base Term: The primary food category
- Facets: The food code in FACETS format
- Detailed/Simple Description: Different descriptions of the food
- Scientific/Common Name: When available
- Vector Similarity: A score indicating vector embedding similarity
- Collection: Where this result came from (detailed, simple, or baseterm collection)

{candidates_text}

Select the top 5 most appropriate food codes that best match the user's query.
For each selected food code, explain your reasoning for why it's a good match in 1-2 sentences.
Return your results in this JSON format:
{{
  "selected_foods": [
    {{
      "rank": 1,
      "candidate_id": 5, 
      "baseterm_name": "Name from candidate",
      "facets": "FACETS code from candidate",
      "reasoning": "Your explanation"
    }},
    ...more selected foods...
  ]
}}
"""

        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)

            # Enhance the result with full candidate information
            for selected in result.get("selected_foods", []):
                candidate_id = selected.get("candidate_id")
                if candidate_id is not None and 1 <= candidate_id <= len(
                    merged_candidates
                ):
                    # Add the full candidate information to the selection (0-indexed)
                    selected["full_info"] = merged_candidates[candidate_id - 1]

            return {
                "query": query_text,
                "all_results": all_results,
                "merged_candidates": merged_candidates,
                "llm_ranked_results": result,
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return {
                "query": query_text,
                "all_results": all_results,
                "merged_candidates": merged_candidates,
                "error": str(e),
            }

    return {
        "query": query_text,
        "all_results": all_results,
        "merged_candidates": merged_candidates,
    }


if __name__ == "__main__":
    # Load both vector databases
    collections = load_all_collections()

    # Interactive query loop
    print("\nFood Database Query Tool")
    print("Type 'exit' to quit")
    print("Type 'switch' to toggle between detailed and simple collections")

    # Default to detailed collection
    current_collection_type = "detailed"
    print(f"Currently using the {current_collection_type} collection")

    while True:
        query = input("\nEnter food description to search: ")

        if query.lower() in ["exit", "quit", "q"]:
            break

        if query.lower() == "switch":
            # Toggle between detailed and simple
            current_collection_type = (
                "simple" if current_collection_type == "detailed" else "detailed"
            )
            print(f"Switched to {current_collection_type} collection")
            continue

        if not query.strip():
            continue

        # Query the current database
        results = query_vector_database(
            query, collections[current_collection_type], n_results=25
        )

        # Display results
        print(format_results(results, current_collection_type))
