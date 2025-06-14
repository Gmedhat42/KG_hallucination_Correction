# The goal of this script is to re-construct the triplets retrived from the KG into a sentence, same for the triplets extracted from the user query

import re 
from typing import Optional

def format_hypothesis_sentence(triple: tuple) -> Optional[str]:
    """
    Converts an (S, P, O) triple into a simple grammatical sentence for hypothesis.

    Args:
        triple (tuple): The (Subject, Predicate, Object) tuple.

    Returns:
        str: The formatted sentence, or None if input is invalid.
    """
    # Input validation
    if not isinstance(triple, tuple) or len(triple) != 3:
        # print(f"  Warning: Cannot format invalid triple: {triple}") # SILENCED
        return None # Indicate formatting failure

    subject, predicate, object_ = triple

    # Basic predicate-to-verb conversion (can be expanded)
    verb_phrase = predicate.replace("_", " ")
    if predicate == "is_type":
        verb_phrase = "is a type of"
    elif predicate == "belongs_to_class":
        verb_phrase = "belongs to class"
    elif predicate == "has_generic_name":
        verb_phrase = "has generic name"
    elif predicate == "has_rx_status":
        verb_phrase = "has Rx status"
    elif predicate == "has_pregnancy_category":
        verb_phrase = "has pregnancy category"
    elif predicate == "has_side_effect" or predicate == "causes":
        verb_phrase = "causes" # Defaulting to 'causes'
    elif predicate == "treats":
         verb_phrase = "treats"
    # Add more specific conversions if needed based on predicates from Step 2

    # Construct the sentence
    try:
        hypothesis_sentence = f"{str(subject)} {verb_phrase} {str(object_)}."
        # Optional: Capitalize first letter
        # hypothesis_sentence = hypothesis_sentence[0].upper() + hypothesis_sentence[1:]
    except Exception as e:
        # Handle potential errors during string formatting (unlikely with basic types)
        print(f" Error formatting sentence for {triple}: {e}")
        return None

    # print(f"  - Formatted Hypothesis: '{hypothesis_sentence}'") # SILENCED
    return hypothesis_sentence

