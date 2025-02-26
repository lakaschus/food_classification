import streamlit as st
import os
import sys

# Add the solution_1_rag_bot directory to the path so we can import from it
sys.path.append(".")
from solution_1_rag_bot.query_vector_database import (
    load_all_collections,
    query_vector_database,
)

# Set page title and configuration
st.set_page_config(page_title="Food Description Lookup", layout="wide")

# App title and description
st.title("Food Description Lookup")
st.markdown(
    """
This app helps you find standardized food codes and descriptions based on your input.
Enter a food description, and the system will search for the closest matches in our database.
"""
)

# Initialize session state for collections if not already done
if "collections" not in st.session_state:
    with st.spinner("Loading food database... This may take a moment."):
        try:
            st.session_state.collections = load_all_collections()
            st.success("Database loaded successfully!")
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            st.stop()

# Create sidebar for settings
st.sidebar.header("Search Settings")
collection_type = st.sidebar.radio(
    "Search Mode:",
    ["detailed", "simple"],
    help="Detailed mode searches comprehensive food descriptions. Simple mode uses more basic descriptions.",
)

num_results = st.sidebar.slider(
    "Number of results to show:",
    min_value=1,
    max_value=10,
    value=3,
    help="How many matching food items to display",
)

# Main input area
food_description = st.text_input(
    "Enter a food description:",
    placeholder="e.g., confectioners' sugar for baking",
    help="Describe the food item you're looking for",
)

# Search button
search_button = st.button("Search")

# Process search when button is clicked
if search_button and food_description:
    with st.spinner("Searching..."):
        # Query the vector database
        results = query_vector_database(
            food_description,
            st.session_state.collections[collection_type],
            n_results=num_results,
        )

        # Display results
        if results["documents"] and results["documents"][0]:
            st.subheader("Search Results")

            # Create columns for better layout
            for i, (document, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Calculate similarity score
                cosine_similarity = 1 - (distance**2) / 2

                # Create an expander for each result
                with st.expander(
                    f"Result {i+1}: {metadata['baseterm_name']} ({cosine_similarity:.2f} similarity)"
                ):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("**Food Code:**")
                        st.code(metadata["facets"], language=None)

                        st.markdown("**Base Term:**")
                        st.write(metadata["baseterm_name"])

                        st.markdown("**Similarity Score:**")
                        st.write(f"{cosine_similarity:.4f}")

                    with col2:
                        st.markdown("**Detailed Description:**")
                        st.write(metadata["detailed_description"])

                        st.markdown("**Simple Description:**")
                        st.write(metadata["simple_description"])

                        if metadata.get("scientific_name"):
                            st.markdown("**Scientific Name:**")
                            st.write(metadata["scientific_name"])

                        if metadata.get("common_name"):
                            st.markdown("**Common Name:**")
                            st.write(metadata["common_name"])
        else:
            st.warning(
                "No matching results found. Try a different description or search mode."
            )

# Add information about the database
st.sidebar.markdown("---")
st.sidebar.markdown("### About the Database")
st.sidebar.markdown(
    """
This app uses a vector database to find food descriptions that match your query.
The database contains standardized food codes and descriptions from the FACETS system.
"""
)

# Footer
st.markdown("---")
st.markdown("Prototype by d-fine")
