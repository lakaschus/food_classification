# Food2ex - Intelligent Food Coding System

A RAG (Retrieval-Augmented Generation) based system for accurately identifying and mapping user food descriptions to standardized food codes within the EFSA Food Classification and Description System (FoodEx2).

## Overview

This project implements an intelligent food coding system that uses vector embeddings and retrieval-augmented generation to match natural language food descriptions to standardized food codes. The system processes the EFSA FoodEx2 food classification data, creating searchable vector databases that allow for semantic similarity matching against user queries.

## Project Structure

The repository contains three main components:

1. **Data Processing** (`data/` directory)
   - Data extraction and transformation from source files
   - Dataset curation and preparation

2. **RAG Bot Implementation** (`solution_1_rag_bot/` directory)
   - Vector database creation and management
   - Query processing and retrieval mechanisms

3. **User Interface** (`streamlit/` directory)
   - Web application for interacting with the system
   - Results visualization and interpretation

## Data Processing Workflow

The system processes food classification data through the following steps:

### 1. Data Extraction

The `data_extraction.py` script reads data from the EFSA Excel source file and performs initial extraction:

- Extracts facet scopes from the "attribute" sheet where attributeType is "catalogue"
- Extracts food terms from the "term" sheet, focusing on APPROVED items
- Creates separate files for base terms and facets
- Performs data validation and cleaning

### 2. Dataset Creation

The `create_dataset.py` script enriches and curates the extracted data:

- Parses FACETS identifier strings into their component parts
- Constructs detailed food descriptions by combining facet and base term information
- Creates simplified food descriptions for easier matching
- Filters out invalid or incomplete entries
- Generates the final curated dataset

The output includes:
- `food2ex_curated_full.csv`: The complete dataset with all valid food items
- `food2ex_curated_sample_1000.csv`: A sample of 1000 random food items for testing

## Vector Database Creation

The `create_vector_database.py` script handles the creation of vector databases for efficient similarity searching:

1. **Text Cleaning**
   - Removes URLs and special formatting
   - Normalizes text for consistent vectorization

2. **Vectorization**
   - Processes food descriptions using OpenAI's text-embedding-3-small model
   - Creates separate collections for different data fields (detailed descriptions, simple descriptions, base terms)
   - Handles batch processing to manage API limits

3. **Metadata Association**
   - Stores comprehensive metadata with each vector, including FACETS codes, base term names, and descriptions

## RAG Bot Query Processing

The `query_vector_database.py` implements the retrieval and ranking functionality:

1. **Loading Collections**
   - Connects to stored vector databases
   - Maintains the same embedding function for consistency

2. **Query Processing**
   - Vectorizes user queries using the same embedding model
   - Searches for semantically similar food descriptions
   - Ranks results by similarity score

3. **Advanced Multi-Collection Search**
   - Combines results from multiple collections for comprehensive matching
   - Uses LLM-based reranking to improve result relevance
   - Identifies the most likely FACETS codes for a given food description

## Novel Food Code Generation

The system also includes functionality to generate appropriate facet expressions for novel foods not present in the existing database (`novel_food_generator.py`):

1. **Food Description Decomposition**
   - Analyzes user-provided food descriptions using LLM
   - Breaks down complex descriptions into distinct meaningful components
   - Identifies key characteristics relevant to food classification

2. **Component-Based Vector Search**
   - Performs vector searches for each decomposed component
   - Retrieves semantically similar base terms from the database
   - Gathers relevant facet expressions from top matches

3. **Contextual Facet Expression Generation**
   - Combines retrieved facet information as context for LLM
   - Generates appropriate facet expressions (e.g., "F01.A0EZN$F02.A0C5P")
   - Provides explanations for selected facet components
   
This approach allows the system to handle novel or fusion foods by decomposing them into known components, finding relevant classifications, and then synthesizing appropriate facet expressions following the FoodEx2 classification system structure.

## Streamlit Web Application

The `streamlit/app.py` provides a user-friendly interface for interacting with the system:

1. **User Input**
   - Accepts natural language food descriptions
   - Provides options for search customization

2. **Results Visualization**
   - Displays matched food items with similarity scores
   - Shows detailed information about FACETS codes and base terms
   - Provides explanations of the food classification components

3. **Interactive Exploration**
   - Allows users to explore the food classification system
   - Provides feedback mechanisms to improve future matches
   
4. **Novel Food Generation**
   - Enables users to describe novel foods not in the existing database
   - Generates appropriate facet expressions for these novel foods
   - Provides detailed analysis of the decomposition and generation process

## Usage

1. **Data Preparation**
   - Run `data_extraction.py` to extract data from the EFSA Excel file
   - Run `create_dataset.py` to create the curated dataset

2. **Vector Database Creation**
   - Run `create_vector_database.py` to create vector embeddings of food descriptions

3. **Web Application**
   - Start the Streamlit app with `streamlit run streamlit/app.py`
   - Enter food descriptions to find matching standardized food codes
   - Use the "Novel Food Code Generator" tab for generating facet expressions for new foods

## Technical Implementation Details

- **Vector Database**: ChromaDB for efficient vector storage and retrieval
- **Embeddings**: OpenAI's text-embedding-3-small model for semantic vectorization
- **LLM Integration**: OpenAI GPT models for advanced reranking and analysis
- **Web Framework**: Streamlit for interactive user interface

Note: OpenAI models have been used for quick development. But for a productive environment we recommend to use a local open weights model for better cost efficiency, faster processing and data security.

## Conclusion

This system demonstrates how modern RAG techniques can be applied to the complex domain of food classification, offering a powerful tool for researchers, nutritionists, and data scientists working with food consumption data. By accurately mapping natural language food descriptions to standardized codes and generating appropriate classifications for novel foods, the system enhances data quality and interoperability across food safety and nutrition studies. 