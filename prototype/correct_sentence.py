# The script is to correct the sentence , if any hallucination was detected by the vectara model

import os
from typing import Optional, Tuple
from langchain_neo4j import Neo4jGraph 


def verify_and_correct_triple(
    graph_client: Neo4jGraph,
    original_triple: tuple,
    is_hallucinated: bool
    ) -> Optional[tuple]:
    """
    Verifies a triple based on hallucination status. If hallucinated, attempts
    correction by querying KG for the correct object. For side effect predicates,
    it replaces the object with the KG's side effect text block.

    Args:
        graph_client (Neo4jGraph): The initialized Neo4j connection object.
        original_triple (tuple): The (Subject, Predicate, Object) triple extracted from LLM.
        is_hallucinated (bool): The result from the hallucination detection step.

    Returns:
        tuple or None: The verified/corrected triple, or None if correction fails.
    """
    if graph_client is None:
        print("  ERROR: Neo4j client not provided to verify_correct_triple.")
        return None
    if not isinstance(original_triple, tuple) or len(original_triple) != 3:
        return None
    if not isinstance(is_hallucinated, bool):
        is_hallucinated = False

    subject, predicate, original_object = original_triple

    if not is_hallucinated:
        return original_triple # Return original, it's verified


    subject_lower = subject.lower()
    params = {'subject_lower': subject_lower}

    predicate_map = {
        "treats":           "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:TREATS]->(o:Condition) RETURN o.name AS correct_object",
        "is_type":          """MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:HAS_GENERIC]->(g:Generic) RETURN g.name AS correct_object UNION MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:BELONGS_TO_CLASS]->(c:Class) RETURN c.name AS correct_object""",
        "belongs_to_class": "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:BELONGS_TO_CLASS]->(o:Class) RETURN o.name AS correct_object",
        "has_generic_name": "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:HAS_GENERIC]->(o:Generic) RETURN o.name AS correct_object LIMIT 1",
        "has_rx_status":    "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.rx_otc IS NOT NULL RETURN s.rx_otc AS correct_object LIMIT 1",
        "has_pregnancy_category": "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.pregnancy_category IS NOT NULL RETURN s.pregnancy_category AS correct_object LIMIT 1",
        "causes":           "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.side_effects_text IS NOT NULL AND s.side_effects_text <> '' RETURN s.side_effects_text AS correct_object LIMIT 1",
        "has_side_effect":  "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.side_effects_text IS NOT NULL AND s.side_effects_text <> '' RETURN s.side_effects_text AS correct_object LIMIT 1",
    }

    query = None
    normalized_predicate = predicate.lower().replace(" ", "_")

    if normalized_predicate not in predicate_map:
        return None 
    query = predicate_map[normalized_predicate]
    try:
        results = graph_client.query(query, params=params)
        # print(f"  - KG Correction Query Results: {results}") # SILENCED

        if results:
            correct_objects = [str(item['correct_object']) for item in results if item.get('correct_object') is not None]
            if not correct_objects:
                return None

            if normalized_predicate in ["causes", "has_side_effect"]:
                final_object = correct_objects[0] # The text block
            elif normalized_predicate in ["belongs_to_class", "treats", "is_type"]:
                 final_object = ", ".join(sorted(list(dict.fromkeys(correct_objects))))
            else: 
                 final_object = correct_objects[0]

            final_triple = (subject, predicate, final_object)
            return final_triple
        else:
            return None 

    except Exception as e:
        print(f"  - Error querying KG for correction: {e}") 
        return None

