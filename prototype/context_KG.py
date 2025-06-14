# The goal of this script is to get the answer from the KG according to each triplet


import os
from langchain_neo4j import Neo4jGraph 


def retrieve_kg_context_triples(graph_client: Neo4jGraph, input_triple: tuple):
    """
    Retrieves ground truth facts from the KG related to the input triple's S/P.
    Returns a LIST of KG fact triples [(S, P, KG_Object), ...] or status message string.

    Args:
        graph_client (Neo4jGraph): Initialized Neo4jGraph connection object.
        input_triple (tuple): The (Subject, Predicate, Object) triple extracted
                              from the LLM output. Used to determine query.

    Returns:
        list or str: A list of ground truth triples found in the KG as
                     [(Subject, Predicate, KG_Object), ...],
                     an empty list [] if facts are not found for existing subject,
                     or a status message string if subject not found or error.
    """
    if graph_client is None:

        return "Error: Neo4j connection not available."
    if not isinstance(input_triple, tuple) or len(input_triple) != 3:
        return "Invalid triple format provided."

    subject, predicate, _ = input_triple 
    subject_lower = subject.lower()
    params = {'subject_lower': subject_lower}
    original_predicate = predicate 


    predicate_map = {
        "treats":           "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:TREATS]->(o:Condition) RETURN o.name AS kg_object",
        "is_type":          """MATCH (s:Drug) WHERE lower(s.name) = $subject_lower OPTIONAL MATCH (s)-[:HAS_GENERIC]->(g:Generic) RETURN g.name AS kg_object UNION MATCH (s:Drug) WHERE lower(s.name) = $subject_lower OPTIONAL MATCH (s)-[:BELONGS_TO_CLASS]->(c:Class) RETURN c.name AS kg_object""",
        "belongs_to_class": "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:BELONGS_TO_CLASS]->(o:Class) RETURN o.name AS kg_object",
        "has_generic_name": "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower MATCH (s)-[:HAS_GENERIC]->(o:Generic) RETURN o.name AS kg_object",
        "has_rx_status":    "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.rx_otc IS NOT NULL RETURN s.rx_otc AS kg_object",
        "has_pregnancy_category": "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.pregnancy_category IS NOT NULL RETURN s.pregnancy_category AS kg_object",
        "causes":           "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.side_effects_text IS NOT NULL AND s.side_effects_text <> '' RETURN s.side_effects_text AS kg_object", # Returns full text block as the object
        "has_side_effect":  "MATCH (s:Drug) WHERE lower(s.name) = $subject_lower AND s.side_effects_text IS NOT NULL AND s.side_effects_text <> '' RETURN s.side_effects_text AS kg_object", # Returns full text block as the object
    }

    query = None
    normalized_predicate = predicate.lower().replace(" ", "_")

    if normalized_predicate in predicate_map:
        query = predicate_map[normalized_predicate]
    else:
        check_exists_query = "MATCH (d:Drug) WHERE lower(d.name) = $subject_lower RETURN count(d) > 0 AS exists"
        try:
             exists_result = graph_client.query(check_exists_query, params=params)
             if exists_result and exists_result[0]['exists']:
                  return f"Subject '{subject}' found, but predicate '{predicate}' is not mapped for context retrieval."
             else:
                  return f"Subject '{subject}' not found in the Knowledge Graph."
        except Exception as e:
             return "Error checking subject existence in KG."

    try:
        results = graph_client.query(query, params=params)
        kg_triples = [] # Initialize list

        if results:
            kg_objects = [str(item['kg_object']) for item in results if item.get('kg_object') is not None]
            if kg_objects:
                for kg_obj in kg_objects:
                    kg_triples.append((subject, original_predicate, kg_obj))
                return kg_triples # <<< Return the list of tuples
            else:
                 return [] 
        else:
            check_exists_query = "MATCH (d:Drug) WHERE lower(d.name) = $subject_lower RETURN count(d) > 0 AS exists"
            exists_result = graph_client.query(check_exists_query, params=params)
            if exists_result and exists_result[0]['exists']:
                 return [] 
            else:
                 return f"Subject '{subject}' not found in the Knowledge Graph."

    except Exception as e:
        return f"Error retrieving context for '{subject}' from KG." 