import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GeoMatcher:
    def __init__(self, anthropic_client=None, model_name="claude-3-5-sonnet-latest"):
        """
        Initializes the GeoMatcher securely utilizing the Anthropic Claude API for semantic evaluations.
        
        Args:
            anthropic_client: An initialized Anthropic client instance mapping credentials natively.
            model_name: The Claude model identifier (e.g., 'claude-3-5-sonnet-latest').
        """
        if anthropic_client is None:
            import anthropic
            import os
            # Generates a dynamic default client if left unspecified manually during instantiation
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            self.client = anthropic_client
            
        self.model_name = model_name

    def match(self, treated_geo_description: str, candidate_geos_list: List[Dict[str, str]]) -> List[str]:
        """
        Dynamically drives an LLM session to evaluate and rank the top-5 most similar candidate markets 
        to a specified treated geo based strictly on encoded textual characteristics.
        
        Args:
            treated_geo_description: A textual description of the treated geo characteristics.
            candidate_geos_list: A list of dicts, where each dict has at least 'geo_id' and 'description'.
                                 Example: [{"geo_id": "NY", "description": "Highly urban, dense..."}, ...]
                                 
        Returns:
            List[str]: A list of up to 5 ranked 'geo_id's that are evaluated as most structurally similar to the treated geo.
        """
        if not candidate_geos_list:
            return []
            
        # 1. Intersect Context into the prompt structure
        prompt_parts = [
            "You are an expert market researcher and data scientist.",
            "Your task is to find the most comparable geographic markets (control group) for a given treated market.",
            "\n### Treated Geo Characteristics:",
            treated_geo_description,
            "\n### Candidate Geos Available:"
        ]
        
        for candidate in candidate_geos_list:
            geo_id = candidate.get("geo_id", "Unknown")
            desc = candidate.get("description", "No description provided.")
            prompt_parts.append(f"- Geo ID: {geo_id}\n  Description: {desc}")
            
        prompt_parts.extend([
            "\n### Task Instructions:",
            "Analyze the demographic, geographic, and economic characteristics of the treated geo and compare them to the candidate geos.",
            "Select and rank the top-5 most similar candidate geos that would serve as the best logical control group.",
            "Provide your reasoning for the selections and the ranking strategy.",
            "\nOutput ONLY valid JSON matching the exact schema below. Do not include any markdown formatting (e.g. ```json), no explanatory text before or after the JSON.",
            "\nExpected JSON Output Schema:",
            "{",
            '  "ranked_geo_ids": ["geo_id_1", "geo_id_2", "geo_id_3", "geo_id_4", "geo_id_5"],',
            '  "reasoning": "Detailed explanation of why this ranking was chosen over other candidates..."',
            "}"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        max_retries = 3
        last_error = None
        
        # 2. Iteratively process payload executing rigorous JSON constraints
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=0.2, # Structured bounds to force deterministic output behavior vs creative drift
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response_text = response.content[0].text.strip()
                
                # Strip out residual Markdown format wrappers Claude natively enforces automatically
                clean_text = response_text
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                    
                clean_text = clean_text.strip()
                
                parsed_data = json.loads(clean_text)
                
                # Guard conditions protecting execution paths mathematically against corrupted hallucinations
                if "ranked_geo_ids" not in parsed_data:
                    raise ValueError("Missing structured 'ranked_geo_ids' array in JSON response.")
                if "reasoning" not in parsed_data:
                    raise ValueError("Missing required 'reasoning' strings in JSON response.")
                    
                # Log the reasoning dynamically 
                logger.info(f"LLM Match Reasoning: {parsed_data['reasoning']}")
                
                ranked_ids = parsed_data["ranked_geo_ids"]
                if not isinstance(ranked_ids, list):
                    raise ValueError("'ranked_geo_ids' structurally must be a strictly resolved list of Strings.")
                    
                # Explicitly bound strictly to top 5 exclusively
                return ranked_ids[:5]
                
            except json.JSONDecodeError as e:
                last_error = f"Failed to parse LLM valid JSON structure cleanly: {e}"
                continue
            except ValueError as e:
                last_error = str(e)
                continue
            except Exception as e:
                # Fatal generic API interruptions
                raise RuntimeError(f"Anthropic API execution framework failed during semantic matching operation: {e}")
                
        raise ValueError(f"Failed to elicit valid target geos reliably from LLM mechanism after {max_retries} loop cycles. Last error trapped: {last_error}")
