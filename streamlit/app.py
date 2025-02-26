import streamlit as st
import os
import sys
import pandas as pd

# Add the solution_1_rag_bot directory to the path so we can import from it
sys.path.append(".")
from solution_1_rag_bot.query_vector_database import (
    load_all_collections,
    query_vector_database,
)


# Function to parse FACETS codes
def parse_facets(facet_string):
    """Parse a FACETS identifier string into its component parts."""
    parts = facet_string.split("#")

    if len(parts) != 2:
        return None, []

    unique_id = parts[0]
    facet_parts = parts[1].split("$")

    facet_baseterm_pairs = []
    for part in facet_parts:
        codes = part.split(".")
        if len(codes) == 2:
            facet_code, baseterm_code = codes
            facet_baseterm_pairs.append((facet_code, baseterm_code))

    return unique_id, facet_baseterm_pairs


# Load facets and base terms data
@st.cache_data
def load_facets_data():
    try:
        facets_df = pd.read_csv("data/facets.csv")
        return facets_df
    except Exception as e:
        st.error(f"Error loading facets data: {str(e)}")
        return None


@st.cache_data
def load_base_terms_data():
    try:
        base_terms_df = pd.read_csv("data/base_terms.csv")
        return base_terms_df
    except Exception as e:
        st.error(f"Error loading base terms data: {str(e)}")
        return None


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

# Load facets and base terms data
facets_df = load_facets_data()
base_terms_df = load_base_terms_data()

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
    ["detailed", "simple", "baseterm"],
    help="Detailed mode searches comprehensive food descriptions. Simple mode uses more basic descriptions.",
)

num_results = st.sidebar.slider(
    "Number of results to show:",
    min_value=1,
    max_value=500,
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
                    # Use a 3-column layout: Components | Description | Additional Info
                    col1, col2 = st.columns([2, 2])

                    with col1:
                        st.markdown("### Food Code Components")
                        st.markdown("*Click on each component for details*")

                        # Parse and display FACETS components in a more visually appealing way
                        if facets_df is not None and "facets" in metadata:
                            # Parse the FACETS code
                            unique_id, facet_baseterm_pairs = parse_facets(
                                metadata["facets"]
                            )

                            if facet_baseterm_pairs:
                                # Display the raw code first with color coding
                                st.markdown("**Raw Food Code:**")

                                # Format the raw code with colors
                                formatted_code = f"{unique_id}#"

                                # Add each facet-baseterm pair with proper coloring
                                for i, (facet_code, baseterm_code) in enumerate(
                                    facet_baseterm_pairs
                                ):
                                    if i > 0:
                                        formatted_code += "$"
                                    formatted_code += f"<span style='color:#1E88E5;font-weight:bold;'>{facet_code}</span>.<span style='color:#D81B60;font-weight:bold;'>{baseterm_code}</span>"

                                # Display the formatted code with word-wrap and width constraints
                                st.markdown(
                                    f"<div style='background-color:#f0f2f6; padding:8px; border-radius:4px; font-family:monospace; word-wrap:break-word; max-width:100%; overflow-wrap:break-word;'>{formatted_code}</div>",
                                    unsafe_allow_html=True,
                                )

                                # Display each component with HTML for visual distinction
                                st.markdown("**Code Breakdown:**")

                                for idx, (facet_code, baseterm_code) in enumerate(
                                    facet_baseterm_pairs
                                ):
                                    # Look up facet information
                                    facet_info = facets_df[
                                        facets_df["code"] == facet_code
                                    ]

                                    # Look up base term information
                                    baseterm_info = None
                                    baseterm_name = ""
                                    baseterm_scope = ""

                                    if base_terms_df is not None:
                                        baseterm_info = base_terms_df[
                                            base_terms_df["termCode"] == baseterm_code
                                        ]
                                        if not baseterm_info.empty:
                                            baseterm_name = baseterm_info.iloc[0][
                                                "termExtendedName"
                                            ]
                                            baseterm_scope = (
                                                baseterm_info.iloc[0]["termScopeNote"]
                                                if "termScopeNote"
                                                in baseterm_info.columns
                                                else ""
                                            )

                                    if not facet_info.empty:
                                        facet_name = facet_info.iloc[0]["name"]
                                        facet_label = facet_info.iloc[0]["label"]
                                        facet_scope = (
                                            facet_info.iloc[0]["scopeNote"]
                                            if "scopeNote" in facet_info.columns
                                            else ""
                                        )

                                        # Create a collapsible section for each component
                                        st.markdown(f"**Component {idx+1}:**")

                                        # Use different colors for facet and baseterm, and include baseterm name
                                        st.markdown(
                                            f"<span style='color:#1E88E5;font-weight:bold;'>{facet_code}</span>.<span style='color:#D81B60;font-weight:bold;'>{baseterm_code}</span> - {facet_label}: {baseterm_name}",
                                            unsafe_allow_html=True,
                                        )

                                        # Use HTML details/summary for collapsible content with base term info
                                        st.markdown(
                                            f"""<details>
                                                <summary>View details</summary>
                                                <p><b>Facet:</b> {facet_name} ({facet_code})<br/>
                                                <b>Facet Description:</b> {facet_scope}<br/>
                                                <b>Base Term:</b> {baseterm_code} - {baseterm_name}<br/>
                                                <b>Base Term Description:</b> {baseterm_scope}</p>
                                                </details>""",
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown("---")
                            else:
                                # Fallback to regular code display if parsing failed
                                st.code(metadata["facets"], language=None)

                    with col2:
                        # Show the simple description first as requested
                        st.markdown("### Food Description")
                        st.markdown("**Simple Description:**")
                        st.write(metadata["simple_description"])

                        # Show additional information
                        st.markdown("### Additional Information")

                        st.markdown("**Base Term:**")
                        st.write(metadata["baseterm_name"])

                        st.markdown("**Similarity Score:**")
                        st.write(f"{cosine_similarity:.4f}")

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
The database contains standardized food codes and descriptions from the FoodEx2 system.
"""
)

# Footer
st.markdown("---")
st.markdown("Prototype by d-fine")
