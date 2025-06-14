# The goal of this script is to extracts a list of triplets from the User Query


import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_triples_from_text(text: str, llm_client: ChatOpenAI):
    """
    Extracts (S, P, O) triples using the provided LLM client.
    Returns list of (Subject, Predicate, Object) tuples.
    """
    if llm_client is None:
        print("Error: LLM Client not provided to extract_triples_from_text.")
        return []


    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert information extraction assistant.
        Extract factual triples (Subject, Predicate, Object) from the following text.
        Focus on claims about drugs, their properties (like class, rx status, pregnancy category), conditions they treat, and side effects they cause.
        Predicates should ideally match or be mappable to the knowledge graph relations: treats, is_type, belongs_to_class, has_rx_status, has_pregnancy_category, has_side_effect, has_generic_name. Use simple verbs if necessary.
        Output each triple on a new line, using only comma separation: Subject, Predicate, Object
        Example: Doxycycline, treats, Acne
        Example: Doxycycline, belongs_to_class, Tetracyclines
        Example: Accutane, has_generic_name, isotretinoin
        Example: Doxycycline, has_rx_status, Rx
        If no clear factual triples are found, output only the word 'None'.
        Do not add explanations or introductory text to your output.
        """),
        ("human", "Text: {input_text}")
    ])

    chain = prompt_template | llm_client | StrOutputParser()

    try:
        raw_triples_output = chain.invoke({"input_text": text})
        # print(f"  LLM Raw Output:\n---\n{raw_triples_output}\n---") # SILENCED

        triples = []
        processed_output = raw_triples_output.strip()
        if processed_output.lower() != 'none' and processed_output:
            for line in processed_output.split('\n'):
                line = re.sub(r'^[\*\-]\s*', '', line).strip()
                parts = [part.strip() for part in line.split(',')]
                if len(parts) == 3:
                    subject = parts[0].strip('". ')
                    predicate = parts[1].strip('". ')
                    object_ = parts[2].strip('". ')
                    if subject and predicate and object_:
                        triples.append((subject, predicate, object_))

        return triples

    except Exception as e:
        print(f"  Error during triple extraction: {e}")
        return []