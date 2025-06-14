# File: mainfinal.py

import os
import re
import time
import json
from dotenv import load_dotenv

from prototype.extract_triplets import extract_triples_from_text
from context_KG import retrieve_kg_context_triples
from prototype.construct_sentence import format_hypothesis_sentence
from prototype.vecatara import detect_hallucinations_batch, hf_model  
from prototype.correct_sentence import verify_and_correct_triple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer
import torch
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_extractor = None
llm_expander = None  # New LLM for query expansion
llm_generator = None  # New LLM for generative correction
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found.")
else:
    try:
        llm_extractor = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)
        llm_expander = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.3)  # Slightly higher temperature for more creative expansions
        llm_generator = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.2)  # Lower temperature for more controlled generation
    except Exception as e:
        print(f"Error initializing OpenAI LLM: {e}")

from langchain_neo4j import Neo4jGraph
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j_graph = None # Initialize variable
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    print("ERROR: Neo4j connection details missing or incomplete.")
else:
    try:
        neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        neo4j_graph.query("RETURN 1")
        print("Neo4j graph client initialized.")
    except Exception as e:
        print(f"ERROR: Failed to connect to Neo4j: {e}")

def expand_query_with_context(input_text: str, graph_client: Neo4jGraph, llm_client: ChatOpenAI) -> tuple:
    """
    Expands the user's query using LLM with context from the knowledge graph.
    Returns both the expanded query and the original query.
    """
    if llm_client is None or graph_client is None:
        return input_text, input_text

    # Get some sample data from the graph to provide context
    try:
        sample_query = """
        MATCH (d:Drug)
        WITH d LIMIT 5
        OPTIONAL MATCH (d)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (d)-[:HAS_GENERIC]->(g:Generic)
        OPTIONAL MATCH (d)-[:BELONGS_TO_CLASS]->(cl:Class)
        RETURN d.name as drug, 
               collect(distinct c.name) as conditions,
               collect(distinct g.name) as generics,
               collect(distinct cl.name) as classes,
               d.pregnancy_category as pregnancy_category,
               d.rx_otc as rx_status
        """
        sample_data = graph_client.query(sample_query)
        
        # Format sample data for the prompt
        sample_text = "\nSample data from knowledge graph:\n"
        for item in sample_data:
            sample_text += f"Drug: {item['drug']}\n"
            if item['conditions']:
                sample_text += f"  Treats: {', '.join(item['conditions'])}\n"
            if item['generics']:
                sample_text += f"  Generic names: {', '.join(item['generics'])}\n"
            if item['classes']:
                sample_text += f"  Drug classes: {', '.join(item['classes'])}\n"
            if item['pregnancy_category']:
                sample_text += f"  Pregnancy category: {item['pregnancy_category']}\n"
            if item['rx_status']:
                sample_text += f"  Rx status: {item['rx_status']}\n"
            sample_text += "\n"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in medical information extraction and query understanding.
            Your task is to expand or clarify the user's query about drugs to make it more specific and extractable.
            Use the sample data from the knowledge graph to understand the available information structure.
            
            Guidelines:
            1. Keep the original meaning and intent
            2. Make implicit information explicit
            3. Add relevant context from the sample data
            4. Ensure the expanded query can be parsed into clear triples
            5. If the query is about pregnancy safety, include the pregnancy category
            6. If the query is about treatment, include the condition being treated
            
            Sample data for context:
            {sample_data}
            
            Original query: {input_text}
            
            Provide an expanded version that maintains the original meaning but is more specific and extractable."""),
            ("human", "Please expand this query to make it more specific while maintaining its original meaning.")
        ])

        chain = prompt_template | llm_client | StrOutputParser()
        
        expanded_text = chain.invoke({
            "input_text": input_text,
            "sample_data": sample_text
        })
        
        return expanded_text, input_text  # Return both expanded and original

    except Exception as e:
        print(f"Error in query expansion: {e}")
        return input_text, input_text  # Return original if expansion fails

def regenerate_text_from_triples(supported_triples, corrected_triples):
    final_triples = supported_triples + corrected_triples
    if not final_triples:
        return "Based on the analysis, no reliable factual statements could be confirmed or constructed."
    triples_by_subject = {}; unique_triples_set = set()
    for triple in final_triples:
        if triple not in unique_triples_set:
            subject = triple[0]
            if subject not in triples_by_subject: triples_by_subject[subject] = []
            triples_by_subject[subject].append(triple)
            unique_triples_set.add(triple)
    all_sentences = []
    for subject in sorted(triples_by_subject.keys()):
        for triple in triples_by_subject[subject]:
            sentence = format_hypothesis_sentence(triple)
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                all_sentences.append(sentence)
    regenerated_text = " ".join(all_sentences)
    return regenerated_text

def generate_correction(original_text: str, correct_triples: list, llm_client: ChatOpenAI) -> str:
    """
    Uses generative AI to create a natural correction based on the correct triples.
    """
    if not correct_triples or llm_client is None:
        return "No corrections could be generated."

    # Format the correct triples for the prompt
    triple_text = "\n".join([f"- {t[0]} {t[1]} {t[2]}" for t in correct_triples])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert medical writer who corrects factual claims about drugs.
        Given the original text and the correct facts, generate a natural, flowing correction.
        The correction should:
        1. Maintain the original writing style and tone
        2. Incorporate all the correct facts naturally
        3. Be clear and professional
        4. Avoid introducing any new claims
        
        Original text: {original_text}
        
        Correct facts:
        {correct_triples}
        
        Generate a corrected version that incorporates these facts naturally."""),
        ("human", "Please correct the text while maintaining its style and incorporating all the correct facts.")
    ])

    chain = prompt_template | llm_client | StrOutputParser()
    
    try:
        corrected_text = chain.invoke({
            "original_text": original_text,
            "correct_triples": triple_text
        })
        return corrected_text
    except Exception as e:
        print(f"Error generating correction: {e}")
        return "Error generating correction."

def generate_hybrid_correction(original_text: str, rule_based_text: str, correct_triples: list, llm_client: ChatOpenAI) -> str:
    """
    Generates a hybrid correction that combines rule-based and generative approaches.
    """
    if not correct_triples or llm_client is None:
        return "No hybrid correction could be generated."

    # Format the correct triples for the prompt
    triple_text = "\n".join([f"- {t[0]} {t[1]} {t[2]}" for t in correct_triples])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert medical writer who combines rule-based and generative approaches to correct factual claims about drugs.
        Given the original text, rule-based correction, and the correct facts, generate a natural, flowing correction that:
        1. Maintains the original writing style and tone
        2. Incorporates all the correct facts naturally
        3. Uses the rule-based correction as a foundation
        4. Adds natural language flow and context
        5. Avoids introducing any new claims
        
        Original text: {original_text}
        
        Rule-based correction: {rule_based_text}
        
        Correct facts:
        {correct_triples}
        
        Generate a corrected version that combines the precision of rule-based correction with natural language flow."""),
        ("human", "Please generate a correction that combines both approaches while maintaining accuracy and natural flow.")
    ])

    chain = prompt_template | llm_client | StrOutputParser()
    
    try:
        corrected_text = chain.invoke({
            "original_text": original_text,
            "rule_based_text": rule_based_text,
            "correct_triples": triple_text
        })
        return corrected_text
    except Exception as e:
        print(f"Error generating hybrid correction: {e}")
        return "Error generating hybrid correction."

def generate_hallucinated_claims(llm_client: ChatOpenAI, graph_client: Neo4jGraph) -> list:
    """
    Generates a list of realistic but potentially hallucinated claims about drugs.
    These claims are designed to be verifiable against our knowledge graph.
    """
    if llm_client is None or graph_client is None:
        return ["Error: LLM client or graph client not available"]

    # Get sample data from the graph to provide context
    try:
        sample_query = """
        MATCH (d:Drug)
        WITH d LIMIT 5
        OPTIONAL MATCH (d)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (d)-[:HAS_GENERIC]->(g:Generic)
        OPTIONAL MATCH (d)-[:BELONGS_TO_CLASS]->(cl:Class)
        RETURN d.name as drug, 
               collect(distinct c.name) as conditions,
               collect(distinct g.name) as generics,
               collect(distinct cl.name) as classes,
               d.pregnancy_category as pregnancy_category,
               d.rx_otc as rx_status
        """
        sample_data = graph_client.query(sample_query)
        
        # Format sample data for the prompt
        sample_text = "\nSample data from our knowledge graph:\n"
        for item in sample_data:
            sample_text += f"Drug: {item['drug']}\n"
            if item['conditions']:
                sample_text += f"  Treats: {', '.join(item['conditions'])}\n"
            if item['generics']:
                sample_text += f"  Generic names: {', '.join(item['generics'])}\n"
            if item['classes']:
                sample_text += f"  Drug classes: {', '.join(item['classes'])}\n"
            if item['pregnancy_category']:
                sample_text += f"  Pregnancy category: {item['pregnancy_category']}\n"
            if item['rx_status']:
                sample_text += f"  Rx status: {item['rx_status']}\n"
            sample_text += "\n"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in generating realistic but potentially incorrect claims about drugs.
            Your task is to generate claims that can be verified against our specific drug knowledge graph.
            
            Here's what we know about our dataset:
            1. We have information about specific drugs, their conditions, classes, and properties
            2. We can verify claims about:
               - Drug treatments and conditions
               - Drug classes
               - Prescription status (Rx vs OTC)
               - Pregnancy categories
               - Generic names
            3. We can only verify claims about drugs and properties in our knowledge graph
            
            {sample_data}
            
            Generate 10 claims that:
            1. Are realistic and sound plausible
            2. Can be verified against our knowledge graph
            3. May contain common misconceptions or incorrect information
            4. Focus ONLY on the drugs and properties shown in the sample data
            5. Use natural, conversational language
            6. Are concise (1-2 sentences each)
            
            Format each claim on a new line, starting with a number and period.
            Example format:
            1. [Drug from sample] is safe to take during pregnancy.
            2. [Drug from sample] is available over the counter for [condition from sample].
            
            IMPORTANT: Only use drugs, conditions, and properties that appear in the sample data.
            Do not make claims about drugs or properties not shown in the sample."""),
            ("human", "Please generate 10 realistic but potentially incorrect claims about drugs from our knowledge graph.")
        ])

        chain = prompt_template | llm_client | StrOutputParser()
        
        claims = chain.invoke({"sample_data": sample_text})
        # Split into list and clean up
        claims_list = [claim.strip() for claim in claims.split('\n') if claim.strip()]
        # Remove numbers and periods from the start
        claims_list = [re.sub(r'^\d+\.\s*', '', claim) for claim in claims_list]
        return claims_list[:10]  # Ensure we only return 10 claims
    except Exception as e:
        print(f"Error generating hallucinated claims: {e}")
        return ["Error generating claims"]

def run_correction_pipeline(input_text: str) -> dict:
    pipeline_results = {
        "original_text": input_text,
        "expanded_text": input_text,
        "extracted_triples": [],
        "verification_details": [],
        "final_triples": [],
        "rule_based_text": "",
        "generative_text": "",
        "hybrid_text": "",
        "status": "Processing",
        "counts": {"extracted": 0, "hallucinated": 0, "corrected": 0,
                   "uncorrected": 0, "supported": 0, "detection_errors": 0,
                   "processing_errors": 0}
    }

    if llm_extractor is None or neo4j_graph is None or hf_model is None:
        pipeline_results["status"] = "Error: Initialization Failed (Components missing)"
        if llm_extractor is None: print("Initialization Error: llm_extractor is None")
        if neo4j_graph is None: print("Initialization Error: neo4j_graph is None")
        if hf_model is None: print("Initialization Error: hf_model is None")
        return pipeline_results

    try:
        # First attempt to extract triples
        extracted_triples = extract_triples_from_text(input_text, llm_extractor)
        
        # If no triples extracted, try query expansion
        if not extracted_triples and llm_expander is not None:
            expanded_text, original_text = expand_query_with_context(input_text, neo4j_graph, llm_expander)
            pipeline_results["expanded_text"] = expanded_text
            extracted_triples = extract_triples_from_text(expanded_text, llm_extractor)
        
        pipeline_results["extracted_triples"] = extracted_triples
        pipeline_results["counts"]["extracted"] = len(extracted_triples)
        
        if not extracted_triples:
            pipeline_results["status"] = "Completed: No triples extracted"
            return pipeline_results
            
    except Exception as e:
        pipeline_results["status"] = "Error: Triple Extraction Failed"
        pipeline_results["counts"]["processing_errors"] += 1
        return pipeline_results

    premises_for_detection = []
    hypotheses_for_detection = []
    triples_to_verify = []

    for i, triple in enumerate(extracted_triples):
        kg_premise_triples_or_status = retrieve_kg_context_triples(neo4j_graph, triple)
        premise_sentence_str = f"Context Error or Subject/Predicate Not Found ({kg_premise_triples_or_status})"
        if isinstance(kg_premise_triples_or_status, list) and kg_premise_triples_or_status:
            premise_sentences = [s for s in [format_hypothesis_sentence(t) for t in kg_premise_triples_or_status] if s]
            if premise_sentences: premise_sentence_str = " ".join(premise_sentences)
            else: premise_sentence_str = f"No specific facts found in KG for {triple[0]} regarding {triple[1]}."
        elif isinstance(kg_premise_triples_or_status, str):
             premise_sentence_str = kg_premise_triples_or_status
        hypothesis_sentence = format_hypothesis_sentence(triple)
        if hypothesis_sentence is None: continue
        premises_for_detection.append(premise_sentence_str)
        hypotheses_for_detection.append(hypothesis_sentence)
        triples_to_verify.append(triple)

    detection_results = []
    if premises_for_detection:
        try:
            detection_results = detect_hallucinations_batch(
                hf_model,
                premises_for_detection,
                hypotheses_for_detection
            )
            pipeline_results["counts"]["detection_errors"] = sum(1 for r in detection_results if r.get('is_hallucinated') is None)
            
            # Print hallucination scores
            print("\nHallucination Scores:")
            print("-" * 50)
            for i, result in enumerate(detection_results):
                print(f"\nTriple {i+1}:")
                print(f"Hypothesis: {hypotheses_for_detection[i]}")
                print(f"Premise: {premises_for_detection[i]}")
                print(f"Vectara Score: {result.get('score', 'N/A'):.4f}")  # Format score to 4 decimal places
                print(f"Is Hallucinated: {result.get('is_hallucinated', 'N/A')}")
                print("-" * 50)
                
        except Exception as e:
             pipeline_results["status"] = "Error: Hallucination Detection Failed"
             pipeline_results["counts"]["processing_errors"] += 1
             return pipeline_results

    supported_triples = []
    corrected_triples_list = []
    final_triples_for_regen = []

    if len(detection_results) != len(triples_to_verify):
         pipeline_results["status"] = "Error: Detection Result Mismatch"
         pipeline_results["counts"]["processing_errors"] += 1
         return pipeline_results

    for i, result_dict in enumerate(detection_results):
        original_triple = triples_to_verify[i]
        is_hallucinated = result_dict.get('is_hallucinated')
        pipeline_results["verification_details"].append(result_dict)
        if is_hallucinated is None: is_hallucinated = False
        final_triple = verify_and_correct_triple(neo4j_graph, original_triple, is_hallucinated)
        if final_triple is not None:
            final_triples_for_regen.append(final_triple)
            if is_hallucinated:
                pipeline_results["counts"]["hallucinated"] += 1
                if final_triple != original_triple:
                    pipeline_results["counts"]["corrected"] += 1
                    corrected_triples_list.append(final_triple)
                else:
                    pipeline_results["counts"]["uncorrected"] += 1
                    supported_triples.append(final_triple)
            else:
                pipeline_results["counts"]["supported"] += 1
                supported_triples.append(final_triple)
        else:
             pipeline_results["counts"]["hallucinated"] += 1
             pipeline_results["counts"]["uncorrected"] += 1

    pipeline_results["final_triples"] = final_triples_for_regen

    try:
        # Generate rule-based correction
        rule_based_text = regenerate_text_from_triples(supported_triples, corrected_triples_list)
        pipeline_results["rule_based_text"] = rule_based_text

        # Generate AI-based correction
        if llm_generator is not None:
            generative_text = generate_correction(input_text, final_triples_for_regen, llm_generator)
            pipeline_results["generative_text"] = generative_text

            # Generate hybrid correction
            hybrid_text = generate_hybrid_correction(input_text, rule_based_text, final_triples_for_regen, llm_generator)
            pipeline_results["hybrid_text"] = hybrid_text

        pipeline_results["status"] = "Completed"
    except Exception as e:
        pipeline_results["status"] = "Error: Text Generation Failed"
        pipeline_results["counts"]["processing_errors"] += 1

    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print(f"Original Input Text:\n\"{pipeline_results['original_text']}\"")
    if pipeline_results['expanded_text'] != pipeline_results['original_text']:
        print(f"Expanded Text:\n\"{pipeline_results['expanded_text']}\"")
    print("-" * 30)
    print("Counts:")
    print(f"  - Triples Extracted: {pipeline_results['counts']['extracted']}")
    print(f"  - Hallucinations Detected: {pipeline_results['counts']['hallucinated']}")
    print(f"  - Facts Supported by KG: {pipeline_results['counts']['supported']}")
    print(f"  - Successfully Corrected: {pipeline_results['counts']['corrected']}")
    print(f"  - Uncorrected Hallucinations: {pipeline_results['counts']['uncorrected']}")
    print(f"  - Detection/Processing Errors: {pipeline_results['counts']['detection_errors'] + pipeline_results['counts']['processing_errors']}")
    print("-" * 30)
    print("Corrections:")
    print(f"Rule-based:\n\"{pipeline_results['rule_based_text']}\"")
    if pipeline_results['generative_text']:
        print(f"\nGenerative:\n\"{pipeline_results['generative_text']}\"")
    if pipeline_results['hybrid_text']:
        print(f"\nHybrid:\n\"{pipeline_results['hybrid_text']}\"")
    print("="*60)
    return pipeline_results

if __name__ == "__main__":
    if llm_extractor is None or neo4j_graph is None or hf_model is None:
        print("\nERROR: Critical components failed to initialize. Exiting.")
        exit()
    test_sentence_1 = "Spironolactone helps with hair loss and is available over the counter"
    results1 = run_correction_pipeline(test_sentence_1)
    all_results = [results1]
    try:
        with open("pipeline_run_results1.json", "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\nDetailed results saved to pipeline_run_results.json")
    except Exception as e:
        print(f"\nError saving results to JSON: {e}")