__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
import os
import sys
import pandas as pd
import zipfile

# Check if vector_db folder exists
if not os.path.exists("vector_db"):
    st.info("Vector database not found. Downloading...")

    # Try to import gdown, install if not available
    try:
        import gdown
    except ImportError:
        import subprocess

        st.warning("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    # Download zip file from Google Drive
    zip_path = "vector_db.zip"
    file_id = "1WynqlwWQPSknj3lHPp7GUnPlid-w-dg5"  # Extract ID from the URL

    try:
        output = gdown.download(id=file_id, output=zip_path, quiet=False)

        if not output:
            st.error(
                "Failed to download the vector database. Please check your internet connection or try again later."
            )
            st.stop()

        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        # Remove the zip file after extraction
        os.remove(zip_path)

        st.success("Vector database downloaded and extracted successfully!")
    except Exception as e:
        st.error(f"Error downloading or extracting the database: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        st.stop()

# Add the solution_1_rag_bot directory to the path so we can import from it
sys.path.append(".")
from solution_1_rag_bot.query_vector_database import (
    load_all_collections,
    query_vector_database,
    advanced_multi_collection_search,
)

# Add this new import
from solution_1_rag_bot.novel_food_generator import NovelFoodCodeGenerator


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
        base_terms_df = pd.read_csv("data/facet_expressions.csv")
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

# Create tabs for different functionalities
tabs = st.tabs(["Food Code Search", "Novel Food Code Generator"])

# TAB 1: Food Code Search
with tabs[0]:
    # Create sidebar for settings
    st.sidebar.header("Search Settings")
    search_mode = st.sidebar.radio(
        "Search Mode:",
        ["Standard RAG", "Advanced RAG (LLM Re-ranking)"],
        help="Standard mode searches a single collection. Advanced mode searches all collections and uses an LLM to rank results.",
    )

    if search_mode == "Standard RAG":
        # Original collection selection code

        # Only show the slider in standard mode
        num_results = st.sidebar.slider(
            "Number of results to show:",
            min_value=1,
            max_value=10,
            value=3,
            help="How many matching food items to display",
        )
    else:
        # No need to select a collection in advanced mode as we use all collections
        st.sidebar.info(
            "Advanced mode automatically searches across all collections and presents the top 5 matches selected by LLM Judge."
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
            if search_mode == "Standard RAG":
                # Original code for standard search
                results = query_vector_database(
                    food_description,
                    st.session_state.collections["baseterm"],
                    n_results=num_results,
                )

                # Original code for displaying results
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

                                        for idx, (
                                            facet_code,
                                            baseterm_code,
                                        ) in enumerate(facet_baseterm_pairs):
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
                                                    base_terms_df["termCode"]
                                                    == baseterm_code
                                                ]
                                                if not baseterm_info.empty:
                                                    baseterm_name = baseterm_info.iloc[
                                                        0
                                                    ]["termExtendedName"]
                                                    baseterm_scope = (
                                                        baseterm_info.iloc[0][
                                                            "termScopeNote"
                                                        ]
                                                        if "termScopeNote"
                                                        in baseterm_info.columns
                                                        else ""
                                                    )

                                            if not facet_info.empty:
                                                facet_name = facet_info.iloc[0]["name"]
                                                facet_label = facet_info.iloc[0][
                                                    "label"
                                                ]
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
                                                        <b>Facet Expression:</b> {baseterm_code} - {baseterm_name}<br/>
                                                        <b>Facet Expression Description:</b> {baseterm_scope}</p>
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

                                st.markdown("**Facet Expression:**")
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
                # Advanced search using all collections
                with st.spinner(
                    "Searching across all collections and analyzing results with LLM..."
                ):
                    advanced_results = advanced_multi_collection_search(
                        food_description,
                        st.session_state.collections,
                        n_results=10,  # Fixed at 10 results per collection
                    )

                    # Display LLM-ranked results if available
                    if (
                        "llm_ranked_results" in advanced_results
                        and "selected_foods" in advanced_results["llm_ranked_results"]
                    ):
                        st.subheader("Top Matches Selected by AI")

                        for i, selected in enumerate(
                            advanced_results["llm_ranked_results"]["selected_foods"]
                        ):
                            rank = selected.get("rank", i + 1)
                            baseterm_name = selected.get("baseterm_name", "")
                            facets = selected.get("facets", "")
                            reasoning = selected.get("reasoning", "")
                            full_info = selected.get("full_info", {})

                            # Get similarity score from full_info
                            similarity = full_info.get("similarity", 0)

                            # Create an expander for each result - using same format as standard search
                            with st.expander(f"Match #{rank}: {baseterm_name}"):
                                # Add the reasoning - this is the only addition to the standard format
                                st.markdown(f"**Why this match?** {reasoning}")
                                st.markdown("---")  # Add separator after reasoning

                                # Use a 2-column layout as in standard search
                                col1, col2 = st.columns([2, 2])

                                with col1:
                                    st.markdown("### Food Code Components")
                                    st.markdown("*Click on each component for details*")

                                    # Parse and display FACETS components in a more visually appealing way
                                    if facets_df is not None and facets:
                                        # Parse the FACETS code
                                        unique_id, facet_baseterm_pairs = parse_facets(
                                            facets
                                        )

                                        if facet_baseterm_pairs:
                                            # Display the raw code first with color coding
                                            st.markdown("**Raw Food Code:**")

                                            # Format the raw code with colors
                                            formatted_code = f"{unique_id}#"

                                            # Add each facet-baseterm pair with proper coloring
                                            for j, (
                                                facet_code,
                                                baseterm_code,
                                            ) in enumerate(facet_baseterm_pairs):
                                                if j > 0:
                                                    formatted_code += "$"
                                                formatted_code += f"<span style='color:#1E88E5;font-weight:bold;'>{facet_code}</span>.<span style='color:#D81B60;font-weight:bold;'>{baseterm_code}</span>"

                                            # Display the formatted code with word-wrap and width constraints
                                            st.markdown(
                                                f"<div style='background-color:#f0f2f6; padding:8px; border-radius:4px; font-family:monospace; word-wrap:break-word; max-width:100%; overflow-wrap:break-word;'>{formatted_code}</div>",
                                                unsafe_allow_html=True,
                                            )

                                            # Display each component with HTML for visual distinction
                                            st.markdown("**Code Breakdown:**")

                                            for idx, (
                                                facet_code,
                                                baseterm_code,
                                            ) in enumerate(facet_baseterm_pairs):
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
                                                        base_terms_df["termCode"]
                                                        == baseterm_code
                                                    ]
                                                    if not baseterm_info.empty:
                                                        baseterm_name = (
                                                            baseterm_info.iloc[0][
                                                                "termExtendedName"
                                                            ]
                                                        )
                                                        baseterm_scope = (
                                                            baseterm_info.iloc[0][
                                                                "termScopeNote"
                                                            ]
                                                            if "termScopeNote"
                                                            in baseterm_info.columns
                                                            else ""
                                                        )

                                                if not facet_info.empty:
                                                    facet_name = facet_info.iloc[0][
                                                        "name"
                                                    ]
                                                    facet_label = facet_info.iloc[0][
                                                        "label"
                                                    ]
                                                    facet_scope = (
                                                        facet_info.iloc[0]["scopeNote"]
                                                        if "scopeNote"
                                                        in facet_info.columns
                                                        else ""
                                                    )

                                                    # Create a collapsible section for each component
                                                    st.markdown(
                                                        f"**Component {idx+1}:**"
                                                    )

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
                                                            <b>Facet Expression:</b> {baseterm_code} - {baseterm_name}<br/>
                                                            <b>Facet Expression Description:</b> {baseterm_scope}</p>
                                                            </details>""",
                                                        unsafe_allow_html=True,
                                                    )
                                                    st.markdown("---")
                                        else:
                                            # Fallback to regular code display if parsing failed
                                            st.code(facets, language=None)

                                with col2:
                                    st.markdown("### Food Description")
                                    st.markdown(
                                        f"**Detailed:** {full_info.get('detailed_description', '')}"
                                    )
                                    st.markdown(
                                        f"**Simple:** {full_info.get('simple_description', '')}"
                                    )

                                    if full_info.get("scientific_name"):
                                        st.markdown(
                                            f"**Scientific Name:** {full_info.get('scientific_name', '')}"
                                        )
                                    if full_info.get("common_name"):
                                        st.markdown(
                                            f"**Common Name:** {full_info.get('common_name', '')}"
                                        )

                                    st.markdown(
                                        f"**Original Vector Similarity:** {full_info.get('similarity', 0):.4f}"
                                    )
                                    st.markdown(
                                        f"**Collection:** {full_info.get('collection', '')}"
                                    )

                    elif "error" in advanced_results:
                        st.error(
                            f"Error in advanced search: {advanced_results['error']}"
                        )
                    else:
                        st.warning("No results found or LLM ranking failed.")

# TAB 2: Novel Food Code Generator
with tabs[1]:
    st.header("Generate Facet Expressions for Novel Foods")
    st.markdown(
        """
    This feature helps you generate facet expressions for novel foods or ingredients that might not exist in the current database.
    Enter a detailed description of your food item, and the system will analyze it and generate appropriate facet expressions.
    """
    )

    # Input for novel food description
    novel_food_description = st.text_area(
        "Enter a detailed description of the novel food or ingredient:",
        height=150,
        help="Provide as much detail as possible about the food item, including ingredients, preparation methods, and characteristics.",
    )

    # Model selection for code generation
    model_options = {
        "GPT-4o mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o",
    }
    selected_model = st.selectbox(
        "Select AI model for facet expression generation:",
        list(model_options.keys()),
        index=0,
        help="GPT-4o provides the most accurate results but may be slower. GPT-4o mini is faster but may be less accurate.",
    )
    model = model_options[selected_model]

    # Detailed results toggle
    show_details = st.checkbox("Show detailed analysis", value=False)

    # Generate button
    if st.button("Generate Facet Expressions"):
        if novel_food_description:
            with st.spinner(
                "Analyzing food description and generating facet expressions..."
            ):
                try:
                    # Initialize the generator
                    generator = NovelFoodCodeGenerator(model=model)

                    # Process the novel food description
                    result = generator.process_novel_food(novel_food_description)

                    # Display the generated result
                    st.subheader("Generated Facet Expressions")
                    st.markdown(result["result"], unsafe_allow_html=False)

                    # Show detailed results if requested
                    if show_details:
                        with st.expander("Detailed Analysis", expanded=True):
                            st.subheader("Food Components")
                            st.json(result["components"])

                            st.subheader("Similar Terms Used for Reference")
                            st.json(result["extracted_terms"])
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a food description.")

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
