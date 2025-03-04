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

        # ONLY load core_terms collection - no more baseterm collection
        self.core_terms_collection = load_vector_database(collection_type="core_terms")

        # Create a mapping of facet codes to facet categories
        self.facet_categories = {}
        for _, row in self.facets_df.iterrows():
            if pd.notna(row.get("code")) and pd.notna(row.get("label")):
                code = row["code"]
                if code.startswith("F"):
                    self.facet_categories[code] = row["label"]

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
        You are an expert food classification assistant. Your task is to decompose the given food description into meaningful food parts.
        
        CONTEXT - Food classification system facets:
        {facet_context}
        
        INSTRUCTIONS:
        1. Analyze the food description and break it down into distinct foods and ingredients
        2. Focus on identifying individual food items, ingredients, and additives
        3. Separate main ingredients from packaging or serving methods
        4. Format your response as a JSON array of strings, where each string is a food item or ingredient
        5. Include common names for foods (e.g., "hummus" instead of "chickpea spread")
        
        FOOD DESCRIPTION: "{food_description}"
        
        OUTPUT FORMAT:
        ```json
        ["food item 1", "food item 2", "ingredient 1", "ingredient 2", ...]
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

    def search_component_in_core_terms(self, component, n_results=5):
        """
        Search for a component in the core_terms collection

        Args:
            component: Food component to search for
            n_results: Number of results to retrieve

        Returns:
            List of matching terms with their codes
        """
        results = query_vector_database(
            component, self.core_terms_collection, n_results=n_results
        )

        matches = []
        if results["documents"][0]:
            for i, (document, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Calculate similarity score
                similarity = 1 - (distance**2) / 2

                term = metadata.get("term", "")
                code = metadata.get("code", "")

                if term and code:
                    matches.append(
                        {
                            "term": term,
                            "code": code,
                            "similarity": similarity,
                            "description": metadata.get("description", ""),
                        }
                    )

        return matches

    def search_all_components(self, components, n_results=5):
        """
        Search for all components in the core_terms collection

        Args:
            components: List of food components
            n_results: Number of results to retrieve per component

        Returns:
            Dictionary mapping components to matches
        """
        component_matches = {}

        for component in components:
            matches = self.search_component_in_core_terms(
                component, n_results=n_results
            )
            if matches:
                component_matches[component] = matches

                # Debug output for each component
                print(f"Component '{component}' matches:")
                for i, match in enumerate(matches[:3]):  # Show top 3
                    print(
                        f"  {i+1}. {match['term']} (Code: {match['code']}, Similarity: {match['similarity']:.4f})"
                    )

        return component_matches

    def generate_food_code(self, food_description, component_matches):
        """
        Generate facet expressions for a novel food based on component matches

        Args:
            food_description: Original food description
            component_matches: Dictionary mapping components to matches from core_terms

        Returns:
            Generated facet expressions and explanation
        """
        # Prepare facet descriptions as context - include ALL facets for proper categorization
        facet_context = self.facets_df[["code", "name", "scopeNote"]].to_string()

        # Format component matches for the prompt
        components_str = ""
        for component, matches in component_matches.items():
            components_str += f"\n{component}:\n"
            for i, match in enumerate(
                matches[:5]
            ):  # Limit to top 5 matches per component
                components_str += f"  {match['term']} (Code: {match['code']}, Similarity: {match['similarity']:.4f})\n"

        # Add facet categories information
        facet_categories_str = "\n".join(
            [f"{code}: {name}" for code, name in self.facet_categories.items()]
        )

        print(f"Facet categories: {facet_categories_str}")

        prompt = f"""
        You are an expert food classification and coding assistant. Your task is to generate facet expressions for a food item.
        
        FOOD DESCRIPTION: "{food_description}"
        
        COMPONENT MATCHES FROM CORE TERMS DATABASE:
        {components_str or "No component matches found"}
        
        FACET CATEGORIES:
        {facet_categories_str}
        
        INSTRUCTIONS:
        1. Analyze the food description and component matches
        2. Identify the MAIN food/ingredient in the description
        3. If there's a high-similarity match (>0.85) for any component, use its code
        4. For components with exact name matches, prioritize using their codes
        5. For "hummus" specifically, use the code A03VN if found
        6. IMPORTANT: All facet expressions MUST be in the format "FXX.XXXXX" where FXX is the facet category
        7. Example: Use "F02.A03ZD" instead of just "A03ZD"
        8. Choose appropriate facet categories from the FACET CATEGORIES list based on what each code represents:
           - F01: Source
           - F02: Part
           - F04: Ingredient
           - F07: Process
           - etc.
        9. Use $ as a separator between multiple facet expressions
        10. Provide a brief explanation of your reasoning
        
        OUTPUT FORMAT:
        ```
        FACET EXPRESSIONS: [generated facet expressions in format F02.A03ZD$F04.A03VN$F07.A07PH]
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
        ONLY using the core_terms collection

        Args:
            food_description: User's novel food description

        Returns:
            Generated facet expressions and explanation
        """
        # Step 1: First try a direct lookup in core_terms collection
        direct_matches = self.search_component_in_core_terms(
            food_description, n_results=10
        )

        # Print debug information for direct matches to the full description
        if direct_matches:
            print(
                f"Found {len(direct_matches)} direct matches for '{food_description}':"
            )
            for i, match in enumerate(direct_matches[:3]):  # Show top 3
                print(
                    f"  {i+1}. {match['term']} (Code: {match['code']}, Similarity: {match['similarity']:.4f})"
                )

        # Check for an exact match with very high similarity
        exact_match = None
        for match in direct_matches:
            if (
                match["term"].lower() == food_description.lower()
                and match["similarity"] > 0.9
            ):
                exact_match = match
                break

        if exact_match:
            # For exact matches, return directly without further processing
            # Make sure the code includes proper facet category if it doesn't have one already
            code = exact_match["code"]
            if not code.startswith("F"):
                # Assume it's a part (F02) if we don't know better
                code = f"F02.{code}"

            explanation = f"The food '{food_description}' exactly matches the term '{exact_match['term']}' in our database with code {code}."
            result = f"""
FACET EXPRESSIONS: {code}
EXPLANATION: {explanation}
"""
            # Include empty extracted_terms for backward compatibility
            return {
                "components": [food_description],
                "direct_match": exact_match,
                "component_matches": {food_description: [exact_match]},
                "extracted_terms": {"baseterm_names": [], "facet_codes": []},
                "result": result,
            }

        # Step 2: If no exact match, decompose and look up each component
        components = self.decompose_food_description(food_description)
        print(f"Food decomposed into {len(components)} components: {components}")

        # Step 3: Search for each component in the core_terms collection ONLY
        component_matches = self.search_all_components(components)

        # Step 4: Generate facet expressions using component matches
        result = self.generate_food_code(food_description, component_matches)

        # Include empty extracted_terms for backward compatibility
        return {
            "components": components,
            "component_matches": component_matches,
            "extracted_terms": {"baseterm_names": [], "facet_codes": []},
            "result": result,
        }
