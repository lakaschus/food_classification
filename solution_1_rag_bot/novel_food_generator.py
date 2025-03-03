import os
import pandas as pd
import sys
from openai import OpenAI
import json

# Add the solution_1_rag_bot directory to the path so we can import from it
sys.path.append(".")
from solution_1_rag_bot.query_vector_database import (
    load_vector_database,
    query_vector_database,
)


class NovelFoodCodeGenerator:
    def __init__(self, openai_api_key=None, model="gpt-4o-mini"):
        """
        Initialize the Novel Food Code Generator

        Args:
            openai_api_key: OpenAI API key (defaults to environment variable)
            model: LLM model to use for generation
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)
        self.model = model
        self.facets_df = pd.read_csv("data/facets.csv")
        self.baseterm_collection = load_vector_database(collection_type="baseterm")

    def decompose_food_description(self, food_description):
        """
        Decompose a food description into meaningful food parts and facet parts

        Args:
            food_description: User's food description

        Returns:
            List of decomposed parts
        """
        # Prepare facet descriptions as context
        facet_context = self.facets_df[["code", "name", "scopeNote"]].to_string()

        # Prompt to decompose the food description
        prompt = f"""
        You are an expert food classification assistant. Your task is to decompose the given food description into meaningful parts.
        
        CONTEXT - Food classification system facets:
        {facet_context}
        
        INSTRUCTIONS:
        1. Analyze the food description and break it down into distinct components
        2. For each component, identify if it's related to any facet (like source, part, process, etc.)
        3. Focus on capturing all meaningful aspects of the food
        4. Format your response as a JSON array of strings, where each string is a descriptive phrase of a component
        
        FOOD DESCRIPTION: "{food_description}"
        
        OUTPUT FORMAT:
        ```json
        ["component 1", "component 2", "component 3", ...]
        ```
        
        Only return the JSON array, nothing else.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            result = response.choices[0].message.content.strip()

            # Extract the JSON array from the response
            import re

            json_match = re.search(r"```json(.*?)```", result, re.DOTALL)
            if json_match:
                result = json_match.group(1).strip()

            # Remove any markdown code block formatting if present
            result = result.replace("```json", "").replace("```", "").strip()

            # Parse the JSON array
            components = json.loads(result)
            return components

        except Exception as e:
            print(f"Error decomposing food description: {str(e)}")
            return [
                food_description
            ]  # Return the original description if decomposition fails

    def search_similar_terms(self, components, n_results=5):
        """
        Search for similar terms in the baseterm collection for each component

        Args:
            components: List of food description components
            n_results: Number of results to retrieve per component

        Returns:
            Dictionary mapping components to search results
        """
        results_dict = {}

        for component in components:
            search_results = query_vector_database(
                component, self.baseterm_collection, n_results=n_results
            )
            results_dict[component] = search_results

        return results_dict

    def extract_facets_and_baseterms(self, results_dict):
        """
        Extract facet expressions and base terms from search results

        Args:
            results_dict: Dictionary of search results for each component

        Returns:
            List of extracted facet expressions and base terms
        """
        extracted_terms = []

        for component, results in results_dict.items():
            if not results["documents"][0]:
                continue

            for i, (document, metadata) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                # Extract baseterm information
                baseterm_name = metadata.get("baseterm_name", "")
                if baseterm_name and baseterm_name not in extracted_terms:
                    extracted_terms.append(baseterm_name)

                # Extract facet information
                facets = metadata.get("facets", "")
                if facets:
                    # Parse facets (assuming format like "A01XD#F01.A0EZN$F02.A0C5P")
                    parts = facets.split("#")
                    if len(parts) > 1:
                        facet_parts = parts[1].split("$")
                        for part in facet_parts:
                            if part and part not in extracted_terms:
                                extracted_terms.append(part)

        return extracted_terms

    def generate_food_code(self, food_description, extracted_terms):
        """
        Generate facet expressions for a novel food based on the food description and extracted terms

        Args:
            food_description: Original food description
            extracted_terms: List of extracted facet expressions and base terms

        Returns:
            Generated facet expressions and explanation
        """
        # Prepare facet descriptions as context
        facet_context = self.facets_df[["code", "name", "scopeNote"]].to_string()

        # Format extracted terms as a string
        extracted_terms_str = "\n".join(extracted_terms)

        prompt = f"""
        You are an expert food classification and coding assistant. Your task is to generate facet expressions for a novel food item.
        
        CONTEXT - Food classification system facets:
        {facet_context}
        
        SIMILAR FOOD CODES AND TERMS:
        {extracted_terms_str}
        
        INSTRUCTIONS:
        1. Analyze the novel food description
        2. Consider the similar food codes and terms provided
        3. Generate ONLY facet expressions (e.g., "F01.A0EZN$F02.A0C5P") without creating a unique ID
        4. Each facet expression should be in the format "FacetCode.BasetermCode" (e.g., "F01.A0EZN")
        5. Use $ as a separator between multiple facet expressions
        6. Provide a brief explanation of your reasoning
        
        NOVEL FOOD DESCRIPTION: "{food_description}"
        
        OUTPUT FORMAT:
        ```
        FACET EXPRESSIONS: [generated facet expressions, e.g., "F01.A0EZN$F02.A0C5P"]
        EXPLANATION: [brief explanation of the facet expressions and reasoning]
        ```
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            print(f"Error generating facet expressions: {str(e)}")
            return f"Error generating facet expressions: {str(e)}"

    def process_novel_food(self, food_description):
        """
        Complete pipeline to process a novel food description and generate facet expressions

        Args:
            food_description: User's novel food description

        Returns:
            Generated facet expressions and explanation
        """
        # Step 1: Decompose the food description
        components = self.decompose_food_description(food_description)

        # Step 2: Search for similar terms for each component
        search_results = self.search_similar_terms(components)

        # Step 3: Extract facet expressions and base terms
        extracted_terms = self.extract_facets_and_baseterms(search_results)

        # Step 4: Generate facet expressions
        result = self.generate_food_code(food_description, extracted_terms)

        return {
            "components": components,
            "extracted_terms": extracted_terms,
            "result": result,
        }
