import os
import json
import pandas as pd
from openai import OpenAI
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
from pathlib import Path
import time

# Added imports for Hugging Face model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error() # Suppress Hugging Face warnings/info

# --- Configuration ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Configure OpenAI client
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in .env file. Please set it.")
    exit()

# OpenAI Model Configuration
OPENAI_MODEL_NAME = "gpt-4.1"
OPENAI_GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.8,
    "max_tokens": 4096,
}

# Constants for Hallucination Detection Model
HALLUCINATION_MODEL_NAME = "vectara/hallucination_evaluation_model"
HALLUCINATION_TOKENIZER_NAME = "google/flan-t5-base" 
HALLUCINATION_THRESHOLD = 0.5 

# --- Helper Functions ---

def load_hf_hallucination_model():
    """Loads the Hugging Face model and tokenizer for hallucination detection."""
    hf_model = None
    # hf_tokenizer = None # Tokenizer loaded but not explicitly used by Vectara's predict
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device for Hugging Face model.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device for Hugging Face model.")
        else:
            device = torch.device("cpu")
            print("Using CPU device for Hugging Face model.")

        print(f"Loading Hugging Face tokenizer: {HALLUCINATION_TOKENIZER_NAME}")
        AutoTokenizer.from_pretrained(HALLUCINATION_TOKENIZER_NAME) # Load to ensure it's cached/available if model needs it implicitly

        print(f"Loading Hugging Face model: {HALLUCINATION_MODEL_NAME}")
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            HALLUCINATION_MODEL_NAME, trust_remote_code=True # trust_remote_code often needed for Vectara
        ).to(device)
        hf_model.eval()
        print("Hugging Face hallucination model initialized successfully.")
        return hf_model
    except Exception as e:
        print(f"ERROR: Failed to initialize Hugging Face hallucination model/tokenizer: {e}")
        return None

def format_triplet_as_sentence(triplet: tuple) -> str:
    """Converts an (S, P, O) triple into a simple grammatical sentence."""
    if not isinstance(triplet, tuple) or len(triplet) != 3:
        return "Invalid triplet for sentence formation."

    subject, predicate, object_ = triplet

    # Clean up predicate for sentence construction
    verb_phrase = predicate.replace("_", " ").lower()
    if verb_phrase.startswith("has property "):
        property_name = verb_phrase.replace("has property ", "")
        # return f"{str(subject)} {property_name} is {str(object_)}." # Alt 1
        return f"{str(subject)} has {property_name} {str(object_)}." # Alt 2
    elif "has " in verb_phrase and not verb_phrase.startswith("has "): # e.g. "sometimes has"
         verb_phrase = verb_phrase # keep as is
    elif verb_phrase.startswith("has "): # e.g. "has genre"
        verb_phrase = verb_phrase # keep as is
    elif verb_phrase == "is type":
        verb_phrase = "is a type of"
    elif "is " not in verb_phrase and "are " not in verb_phrase and "causes " not in verb_phrase and "treats " not in verb_phrase: # basic verb check
        verb_phrase = f"is related to {predicate} for" # Fallback for unknown predicates

    # Simple sentence construction
    return f"{str(subject)} {verb_phrase} {str(object_)}."

def run_hallucination_detection_batch(
    model, # Expect the loaded Hugging Face model object
    premise_sentences: list[str],
    hypothesis_sentence: str # Single hypothesis sentence to compare against all premises
    ) -> list[dict]:
    """
    Runs hallucination detection comparing one hypothesis sentence against multiple premise sentences.
    Returns a list of detection results, one for each premise-hypothesis pair.
    """
    if model is None:
        print("ERROR: Hallucination detection model not initialized for batch.")
        return [{"premise": p, "hypothesis": hypothesis_sentence, "consistency_score": None, "is_hallucinated": None, "error": "Model not loaded"} for p in premise_sentences]

    if not premise_sentences: # If no premises, nothing to compare against
        # print("Warning: No premise sentences provided for hallucination detection.")
        return [] 
    if not hypothesis_sentence:
        print("ERROR: Hypothesis sentence is empty for hallucination detection.")
        return [{"premise": p, "hypothesis": "", "consistency_score": None, "is_hallucinated": None, "error": "Hypothesis empty"} for p in premise_sentences]

    # Create pairs: (premise, hypothesis)
    # The hypothesis sentence is repeated for each premise it's compared against.
    pairs = [(prem, hypothesis_sentence) for prem in premise_sentences]
    
    detection_results = []
    try:
        # print(f"Running hallucination prediction on {len(pairs)} pairs...") # Debug
        with torch.no_grad():
            scores_tensor = model.predict(pairs) # Vectara model's specific predict method
        
        scores = scores_tensor.cpu().numpy().tolist() # Detach, move to CPU, convert

        if len(scores) != len(pairs):
            print(f"ERROR: Mismatch between input pairs ({len(pairs)}) and output scores ({len(scores)}) for hallucination detection.")
            return [{"premise": p, "hypothesis": h, "consistency_score": None, "is_hallucinated": None, "error": "Prediction output mismatch"} for p, h in pairs]

        for i, (prem_sent, hypo_sent) in enumerate(pairs):
            consistency_score = scores[i]
            is_hallucinated = None
            current_error = None

            if not isinstance(consistency_score, (int, float)):
                # print(f"Warning: Invalid score type received for pair {i}: {consistency_score}")
                is_hallucinated = None 
                consistency_score = None
                current_error = "Invalid score type"
            else:
                is_hallucinated = consistency_score < HALLUCINATION_THRESHOLD

            detection_results.append({
                "premise_sentence": prem_sent,
                "hypothesis_sentence": hypo_sent,
                "consistency_score": consistency_score,
                "is_hallucinated": is_hallucinated,
                "error": current_error
            })
    except Exception as e:
        print(f"  - Error during hallucination model.predict batch: {e}")
        return [{"premise": p, "hypothesis": h, "consistency_score": None, "is_hallucinated": None, "error": str(e)} for p, h in pairs]
    
    # print("Hallucination detection batch completed for one hypothesis.") # Debug
    return detection_results

def call_openai_llm(system_prompt: str, user_prompt: str, client: OpenAI, model_name: str, generation_config: dict):
    """Calls the OpenAI API and returns the text response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=generation_config.get("temperature"),
            top_p=generation_config.get("top_p"),
            max_tokens=generation_config.get("max_tokens")
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            print("Warning: OpenAI response was empty or in an unexpected format.")
            return ""
    except Exception as e:
        print(f"ERROR: OpenAI API call failed: {e}")
        print(f"  User Prompt that caused error: {user_prompt[:500]}...")
        return None

def get_neo4j_driver(uri, username, password):
    """Establishes connection to Neo4j and returns a driver object."""
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
        driver.verify_connectivity()
        print("Neo4j driver initialized successfully.")
        return driver
    except Exception as e:
        print(f"ERROR: Failed to connect to Neo4j: {e}")
        return None

def run_cypher_query(driver, query, params=None, database="neo4j", write_query=False):
    """Runs a Cypher query against Neo4j."""
    if params is None:
        params = {}
    try:
        with driver.session(database=database) as session:
            if write_query:
                session.run(query, params)
                return [] 
            else:
                results = session.run(query, params)
                return [record.data() for record in results]
    except Exception as e:
        print(f"ERROR: Cypher query failed: {e}")
        print(f"  Query: {query}")
        if params:
            print(f"  Params: {params}")
        return None

def clear_database(driver, database="neo4j"):
    """Clears all nodes and relationships from the specified Neo4j database."""
    print(f"WARNING: Clearing all data from Neo4j database '{database}' in 3 seconds...")
    time.sleep(3)
    try:
        run_cypher_query(driver, "MATCH (n) DETACH DELETE n", database=database, write_query=True)
        print(f"Database '{database}' cleared.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to clear database: {e}")
        return False

def load_and_prepare_csv_data(csv_filepath_str: str, sample_rows=10):
    """Loads CSV, extracts filename, columns, and sample data.
    The returned URI is simplified to file:///filename.csv for Neo4j LOAD CSV,
    assuming the user has placed the file in Neo4j's import directory.
    The csv_filepath_str is used by pandas to read the file locally."""
    try:
        # This path is for pandas to read the file
        pandas_csv_path = Path(csv_filepath_str)
        if not pandas_csv_path.is_file():
            print(f"ERROR: CSV file not found at the path provided for pandas: {csv_filepath_str}")
            return None, None, None, None

        df = pd.read_csv(pandas_csv_path)
        
        # This filename is extracted from the pandas path and used for the Neo4j URI
        # It's assumed a file with this exact name is in Neo4j's import directory.
        base_filename_for_neo4j = pandas_csv_path.name
        
        columns = df.columns.tolist()
        
        sample_data_df = df.head(sample_rows).where(pd.notnull(df.head(sample_rows)), None)
        sample_data_list_of_dicts = sample_data_df.to_dict(orient='records')

        print(f"CSV '{base_filename_for_neo4j}' (read by pandas from '{csv_filepath_str}') loaded. Columns: {columns}")
        
        # Construct the simplified URI for Neo4j's LOAD CSV
        neo4j_csv_uri = f"file:///{base_filename_for_neo4j}" 
        print(f"Using URI for Neo4j LOAD CSV: {neo4j_csv_uri}")

        # Return the base_filename_for_neo4j as the 'filename' identifier
        return base_filename_for_neo4j, columns, sample_data_list_of_dicts, neo4j_csv_uri
    except Exception as e:
        print(f"ERROR: Failed to load or process CSV: {e}")
        return None, None, None, None

def parse_llm_cypher_output(cypher_string: str) -> list:
    """Parses Cypher string from LLM into a list of individual statements."""
    if not cypher_string:
        return []
    # Remove ```cypher and ``` markers if present
    if cypher_string.strip().startswith("```cypher"):
        cypher_string = cypher_string.strip()[len("```cypher"):]
    if cypher_string.strip().endswith("```"):
        cypher_string = cypher_string.strip()[:-len("```")]
    
    statements = [stmt.strip() for stmt in cypher_string.split(';') if stmt.strip()]
    # Further split by newlines if semicolons are not used for all statements
    final_statements = []
    for stmt in statements:
        final_statements.extend(s.strip() for s in stmt.split('\\n') if s.strip()) # Handle escaped newlines
    
    # A more robust split considering statements might not end with ';' but be on new lines
    # This is tricky as Cypher can have multi-line strings or comments.
    # For now, rely on ';' as primary separator, or significant newlines.
    # A simple approach: split by ';', then for each part, split by newline if it seems to be multiple commands.
    # This assumes the LLM generally separates distinct commands.
    
    # Let's try a cleaner approach based on semicolon or newline as statement terminators,
    # being careful not to split inside string literals (though simple split won't handle this perfectly)
    raw_statements = cypher_string.split(';')
    cleaned_statements = []
    for raw_stmt in raw_statements:
        parts = raw_stmt.split('\\n') # if LLM uses escaped newlines
        for part in parts:
            part = part.strip()
            if part:
                cleaned_statements.append(part)
    
    # If still minimal splitting, try splitting by newline more directly as a fallback
    if len(cleaned_statements) <= 1 and ';' not in cypher_string:
        cleaned_statements = [stmt.strip() for stmt in cypher_string.splitlines() if stmt.strip()]
        
    return cleaned_statements if cleaned_statements else [cypher_string.strip()] # return original if split fails

# --- Main Workflow ---
def main_pipeline():
    # Initialize OpenAI Client
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"OpenAI client initialized for model '{OPENAI_MODEL_NAME}'.")
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return

    # Initialize Hugging Face Hallucination Model
    hf_hallucination_model = load_hf_hallucination_model()
    if hf_hallucination_model is None:
        print("Warning: Hallucination detection model failed to load. Proceeding without it.")
        # Optionally, could exit if this model is critical:
        # return

    # Initialize lists for final output
    all_hypothesis_triplets_final = []
    all_premise_triplets_final = []

    # 1. Setup & CSV Ingestion
    csv_filepath = input("Enter the path to your CSV file (e.g., data/your_data.csv): ")
    filename, columns, sample_data, csv_file_uri = load_and_prepare_csv_data(csv_filepath)

    if not filename:
        return

    # Initialize Neo4j Driver
    neo4j_driver = get_neo4j_driver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    if not neo4j_driver:
        return
    
    generated_cypher_queries_string_for_schema = "" # Initialize
    construct_kg = False # Flag to determine if we should run Step 2 (KG Construction)

    # Confirm before clearing DB
    if input("Clear existing Neo4j database before proceeding? (yes/no): ").lower() == 'yes':
        if not clear_database(neo4j_driver):
            if input("Continue despite database clear failure? (yes/no): ").lower() != 'yes':
                neo4j_driver.close()
                return
        # If DB is cleared, we must generate Cypher and construct the KG
        print("\nDatabase cleared. Proceeding to generate new Cypher schema and construct KG.")
        construct_kg = True
    else:
        print("\nSkipping database clear. Attempting to use existing KG schema or generate if not found.")
        try:
            with open("generated_graph_cypher.txt", "r", encoding="utf-8") as f:
                generated_cypher_queries_string_for_schema = f.read()
            if generated_cypher_queries_string_for_schema and generated_cypher_queries_string_for_schema.strip():
                print("Successfully loaded existing Cypher schema from generated_graph_cypher.txt.")
                print("Skipping LLM1 call for Cypher generation and KG construction step.")
                construct_kg = False # Do not reconstruct if schema is loaded and user said no to clear
            else:
                print("generated_graph_cypher.txt is empty. Proceeding to generate new Cypher schema with LLM1.")
                construct_kg = True # Need to generate schema
        except FileNotFoundError:
            print("generated_graph_cypher.txt not found. Proceeding to generate new Cypher schema with LLM1.")
            construct_kg = True # Need to generate schema
        except IOError as e:
            print(f"Error reading generated_graph_cypher.txt: {e}. Proceeding to generate new Cypher schema.")
            construct_kg = True # Need to generate schema

    if construct_kg and not generated_cypher_queries_string_for_schema: # Only call LLM1 if we need to generate and don't have schema yet
        # 2. LLM Call 1: Knowledge Graph Cypher Generation
        print("\n--- Step 1: Generating Knowledge Graph Cypher (LLM1) ---")
        llm1_system_prompt = "You are a Neo4j Cypher query expert. Generate Cypher queries to create a knowledge graph from this dataset."
        llm1_user_prompt = f"""Dataset Information:
- CSV filename: {filename} # This is base_filename_for_neo4j from load_and_prepare_csv_data
- Columns: {', '.join(columns)}
- Sample data (first few rows):
{json.dumps(sample_data, indent=2)}

Required Cypher Queries:
Your primary goal is to generate queries similar in structure to the example below, but adapted for the provided dataset ({filename}, {columns}, {sample_data}).
IMPORTANT: For the `LOAD CSV` query, use the exact placeholder `{{csv_file_uri_placeholder}}` for the file URI. The script will replace this with, for example, `file:///your_actual_filename.csv`.
Example: `LOAD CSV WITH HEADERS FROM '{{csv_file_uri_placeholder}}' AS row`

Example structure for a TV show dataset (you must adapt this logic for the *actual* dataset provided above):
```cypher
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Show) REQUIRE s.title IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE;

LOAD CSV WITH HEADERS FROM '{{csv_file_uri_placeholder}}' AS row
WITH row
// Create/Merge nodes for primary entities
MERGE (s:Show {{title: row.title}}) // Adapt 'Show' and 'title' to the main entity and its identifier in the provided data
SET s.year = toInteger(row.year), // Adapt properties based on columns
    s.certificate = row.certificate,
    s.duration_min = toInteger(row.duration_min),
    s.rating = toFloat(row.rating),
    s.votes = toInteger(row.votes)
WITH s, row // Ensure 'row' (or main entity variable like 's') is carried if multiple UNWINDs are needed
// Handle multi-valued columns by splitting and creating related nodes/relationships
// Example for a 'genre' column with '|' separation:
UNWIND split(row.genre, '|') AS genre_name_raw
WITH s, row, trim(genre_name_raw) AS genre_name // Carry necessary variables (s, row)
WHERE genre_name <> ''
MERGE (g:Genre {{name: trim(genre_name)}}) // Adapt 'Genre' and 'name'
MERGE (s)-[:HAS_GENRE]->(g) // Adapt relationship type
WITH s, row // Carry necessary variables for next UNWIND
// Example for a 'stars' column with '|' separation:
UNWIND split(row.stars, '|') AS star_name_raw
WITH s, row, trim(star_name_raw) AS star_name // Carry necessary variables
WHERE star_name <> ''
MERGE (a:Actor {{name: trim(star_name)}}) // Adapt 'Actor' and 'name'
MERGE (s)-[:STARS_ACTOR]->(a) // Adapt relationship type

// Example for creating an index (if not covered by unique constraint):
CREATE INDEX IF NOT EXISTS FOR (s:Show) ON (s.year);
```

Please analyze the provided dataset ({filename}, {columns}, {sample_data}) and generate:
1. Appropriate `CREATE CONSTRAINT` statements for unique properties of the main entities you identify.
2. A `LOAD CSV` query to populate the graph. This query should:
   - Use `LOAD CSV WITH HEADERS FROM '{{csv_file_uri_placeholder}}' AS row`.
   - `MERGE` primary entity nodes.
   - `SET` their properties, converting to appropriate types (e.g., `toInteger(row.col)`, `toFloat(row.col)`). Check sample data for type hints.
   - If any columns appear to contain multiple values (e.g., separated by '|', ',', ';'), use `UNWIND split(...)` to create separate nodes for these values (e.g., for tags, categories, multiple authors) and link them to the main entity with appropriate relationships. Ensure `row` or the main entity variable is carried through `WITH` clauses if needed for subsequent UNWINDs.
   - Ensure property names in Cypher match column names from the CSV, or are logically derived and clearly mapped.
3. Any useful `CREATE INDEX` statements for frequently queried properties (other than those covered by unique constraints).

Return ONLY the Cypher queries, with each statement on a new line or separated by semicolons if appropriate. No additional text or explanation."""

        temp_generated_cypher = call_openai_llm(
            llm1_system_prompt,
            llm1_user_prompt,
            openai_client,
            OPENAI_MODEL_NAME,
            OPENAI_GENERATION_CONFIG
        )

        if not temp_generated_cypher:
            print("ERROR: LLM1 failed to generate Cypher queries. Exiting.")
            neo4j_driver.close()
            return
        
        generated_cypher_queries_string_for_schema = temp_generated_cypher
        print("\nLLM1 Generated Cypher (raw):\n", generated_cypher_queries_string_for_schema)
        
        try:
            with open("generated_graph_cypher.txt", "w", encoding="utf-8") as f:
                f.write(generated_cypher_queries_string_for_schema)
            print("Generated Cypher saved to generated_graph_cypher.txt")
        except IOError as e:
            print(f"Error saving Cypher to file: {e}")
    elif not construct_kg and generated_cypher_queries_string_for_schema:
        print("\n--- Step 1: Using existing Cypher Schema from file ---")
        # Schema is already loaded in generated_cypher_queries_string_for_schema
        pass # No LLM1 call needed
    else: # Should not happen if logic above is correct, but as a fallback
        print("ERROR: Inconsistent state for Cypher schema generation. Exiting.")
        neo4j_driver.close()
        return

    if construct_kg:
        # 3. Knowledge Graph Construction in Neo4j
        print("\n--- Step 2: Constructing Knowledge Graph in Neo4j ---")
        
        final_cypher_string_for_construction = generated_cypher_queries_string_for_schema.replace("{csv_file_uri_placeholder}", csv_file_uri)
        cypher_statements = parse_llm_cypher_output(final_cypher_string_for_construction)
        
        constraints_queries = [q for q in cypher_statements if "CREATE CONSTRAINT" in q.upper()]
        load_csv_queries = [q for q in cypher_statements if "LOAD CSV" in q.upper()]
        index_queries = [q for q in cypher_statements if "CREATE INDEX" in q.upper() and "LOAD CSV" not in q.upper()]
        other_queries = [q for q in cypher_statements if q not in constraints_queries and q not in load_csv_queries and q not in index_queries]

        execution_order = constraints_queries + load_csv_queries + index_queries + other_queries

        for i, stmt in enumerate(execution_order):
            print(f"Executing Cypher statement {i+1}/{len(execution_order)}: {stmt[:100]}...")
            result = run_cypher_query(neo4j_driver, stmt, write_query=True)
            if result is None:
                print(f"ERROR: Failed to execute Cypher statement. KG construction may be incomplete.")
                if "LOAD CSV" in stmt.upper():
                    print("LOAD CSV statement failed. Aborting further KG construction.")
                    # Decide if we should exit or allow user to proceed with potentially incomplete KG for validation
                    if input("LOAD CSV failed. Continue with validation against potentially incomplete/empty KG? (yes/no): ").lower() != 'yes':
                        neo4j_driver.close()
                        return
                    break # Break from executing further statements in this KG construction block
            time.sleep(0.5)
        print("Knowledge Graph construction/update attempted.")
    else:
        print("\n--- Step 2: Skipped KG construction as per user choice or existing schema ---")

    # 4. Evolving Context Management & User Input
    llm_context = {
        'csv_filename': filename, # This is base_filename_for_neo4j from load_and_prepare_csv_data
        'csv_columns': columns,
        'data_sample': sample_data,
        'generated_kg_cypher_schema': generated_cypher_queries_string_for_schema 
    }

    user_sentence = input("\n--- Step 3: User Input for Hallucination Check ---\nEnter a sentence to validate against the KG: ")
    if not user_sentence:
        print("No sentence provided. Exiting.")
        if neo4j_driver: neo4j_driver.close()
        return

    # 5. LLM Call 2: Hypothesis Triplet Extraction
    print("\n--- Step 4: Extracting Hypothesis Triplets (LLM2) ---")
    llm2_system_prompt = """You are an expert at extracting factual triplets (Subject, Predicate, Object) from text.
You will receive:
1. A user query/sentence.
2. Sample data from the original CSV dataset.
3. The Cypher queries that were used to define the knowledge graph structure.

Your task:
1. Analyze the 'Cypher Queries (Knowledge Graph Structure)' to understand:
   - Node labels (e.g., `:Show`, `:Genre` from the example, adapt to the actual schema provided).
   - Relationship types (e.g., `:HAS_GENRE`, `:STARS_ACTOR`).
   - Property names for each node/relationship (e.g., `title`, `year`, `name`).
2. Extract factual triplets from the 'User Query'.
3. Each triplet MUST reflect a potential fact that could exist in the knowledge graph defined by the Cypher.
   - For relationships: Subject, RELATIONSHIP_TYPE, Object (e.g., "The Matrix", "HAS_GENRE", "Sci-Fi")
   - For properties: Entity, HAS_PROPERTY_PropertyName, Value (e.g., "The Matrix", "HAS_PROPERTY_year", "1999")
     - Ensure 'PropertyName' is the actual property key from the schema (e.g. if schema has `s.year`, then Predicate is `HAS_PROPERTY_year`).
4. Format each triplet as: Subject, Predicate, Object (comma-separated).
5. Only include triplets that seem plausible given the graph schema.
6. Use the exact labels for Node types (implicitly, as Subjects/Objects will be instances of these) and exact Relationship Types or "HAS_PROPERTY_propertyKey" for Predicates.

Rules for Output:
- Each line should contain exactly one triplet: Subject, Predicate, Object
- If no valid triplets can be extracted based on the schema, return only the single word 'None' (with no other characters or newlines).
- Do not add any explanations or introductory text to your output.

Example format for relationships (if schema has `(s:Show)-[:HAS_GENRE]->(g:Genre)`):
The Matrix, HAS_GENRE, Sci-Fi

Example format for properties (if schema has `s:Show {{year: 1999}}`):
The Matrix, HAS_PROPERTY_year, 1999

Return only the triplets, one per line, or 'None'.
"""
    llm2_user_prompt = f"""User Query: {user_sentence}

Sample Data (from CSV):
{json.dumps(llm_context['data_sample'], indent=2)}

Cypher Queries (Knowledge Graph Structure - this defines available Nodes, Relationships, Properties):
```cypher
{llm_context['generated_kg_cypher_schema']}
```
"""
    hypothesis_triplets_str_raw = call_openai_llm(
        llm2_system_prompt,
        llm2_user_prompt,
        openai_client,
        OPENAI_MODEL_NAME,
        OPENAI_GENERATION_CONFIG
    )
    print("\nLLM2 Generated Triplets (raw):\n", repr(hypothesis_triplets_str_raw)) # Use repr to see exact string

    hypothesis_triplets = []
    if hypothesis_triplets_str_raw and hypothesis_triplets_str_raw.strip().lower() != 'none':
        # First, replace escaped newlines with actual newlines, then split by actual newlines
        processed_output_str = hypothesis_triplets_str_raw.strip().replace('\\\\n', '\n') # Corrected to replace literal '\n'
        lines = processed_output_str.splitlines() # Robustly splits by various newline characters

        for line in lines:
            line = line.strip() # Strip whitespace from individual line
            if not line: continue
            
            parts = [p.strip() for p in line.split(',', 2)] # Split by comma, max 2 splits for 3 parts
            if len(parts) == 3:
                # Further check: ensure no part itself contains an unescaped newline
                # that might indicate a parsing error from a previous step or malformed LLM output
                if any('\n' in part for part in parts):
                    print(f"Warning: Parsed triplet part contains newline after initial split, potential mis-parsing of line: '{line}' -> {parts}")
                    # This might indicate LLM isn't strictly following one triplet per line with only comma separation internal to triplet
                
                hypothesis_triplets.append(tuple(parts))
            else:
                print(f"Warning: Could not parse triplet line '{line}' into 3 parts (got {len(parts)} parts: {parts})")
    
    if not hypothesis_triplets:
        print("No hypothesis triplets extracted by LLM2 or 'None' returned.")
    else:
        print(f"Extracted {len(hypothesis_triplets)} hypothesis triplets:")
        for ht in hypothesis_triplets:
            print(f"  - {ht}")

    llm_context['hypothesis_triplets_str_for_llm3'] = "Extracted Hypothesis Triplets from user query:\n" + \
                                             "\n".join([f"({s},{p},{o})" for s, p, o in hypothesis_triplets]) if hypothesis_triplets else "No hypothesis triplets extracted."
    llm_context['hypothesis_triplets_list'] = hypothesis_triplets # for direct use
    # Populate all_hypothesis_triplets_final
    if hypothesis_triplets:
        all_hypothesis_triplets_final.extend(hypothesis_triplets)

    # 6. LLM Call 3 (Iterative): Validation Cypher Query Generation & Premise Retrieval
    #    & Hallucination Detection
    print("\n--- Step 5: Validating Triplets against KG & Detecting Hallucinations (LLM3, LLM4, HF Model) ---")
    validation_results = [] 
    actual_kg_triplets_map = {} 
    
    # New lists to aggregate all hallucinations before final correction
    all_hallucinated_triplets = []
    all_correction_triplets = []

    if not hypothesis_triplets:
        print("No hypothesis triplets to validate.")
    else:
        for i, hyp_triplet in enumerate(hypothesis_triplets): # Added index for mapping
            subject, predicate, object_ = hyp_triplet
            print(f"\nValidating hypothesis triplet {i+1}/{len(hypothesis_triplets)}: ({subject}, {predicate}, {object_})")
            
            # Initialize storage for LLM4 results for this specific hypothesis
            current_llm4_correction_query_str = None
            actual_kg_trips_found_by_llm4 = []

            # Format hypothesis for Hallucination model
            hypothesis_sentence_for_hf = format_triplet_as_sentence(hyp_triplet)
            print(f"  Formatted Hypothesis Sentence: '{hypothesis_sentence_for_hf}'")

            # Modified prompt structure for OpenAI
            llm3_system_prompt = """You are a Neo4j Cypher query expert. You will receive:
1. A single triplet (Subject, Predicate, Object) to validate.
2. Sample data from the original CSV dataset (for context on data types and typical values).
3. The original Cypher queries that defined the knowledge graph structure (for node labels, relationship types, and property keys).

Your task:
1. Analyze the 'Original Cypher Queries' to understand node labels (e.g., `:Show`), relationship types (e.g., `:HAS_GENRE`), and property keys (e.g., `title`, `year`).
2. Generate a specific Cypher `MATCH` query to find evidence for the given 'Triplet to query for'.
3. If the Predicate is a relationship type (e.g., `HAS_GENRE`, `STARS_ACTOR` - check the schema for actual types):
   - The query should `MATCH (s)-[r:{predicate}]->(o)`.
   - Infer node labels for `s` and `o` based on the relationship type and the overall schema.
   - `s` should match the `Subject` (usually by a name, title, or ID property identified from the schema).
   - `o` should match the `Object_` (similarly, by a name or ID property).
   - The query should `RETURN s.<id_prop> AS subject, type(r) AS predicate, o.<id_prop> AS object`.
     You MUST determine the correct `<id_prop>` (e.g., `title`, `name`, `id`) for nodes `s` and `o` by looking at the `CREATE CONSTRAINT` or `MERGE` clauses in the 'Original Cypher Queries'.
4. If the Predicate is a property assignment (e.g., `HAS_PROPERTY_year`, `HAS_PROPERTY_rating`):
   - The `Predicate` format is 'HAS_PROPERTY_<property_key>'. Extract `<property_key>` (e.g., 'year').
   - The query should `MATCH (s)` where `s` is the node type associated with the `Subject`.
   - `s` should match the `Subject` (usually by its identifying property).
   - The query must check `s.<property_key> = <value_of_Object_>`.
     - Ensure `<value_of_Object_>` is correctly typed (e.g., `toInteger('{object_}')`, `toFloat('{object_}')`, or a string `'{object_}'`). Infer type from schema or sample data.
   - The query should `RETURN s.<id_prop> AS subject, '{predicate}' AS predicate, s.<property_key> AS object`.
     Again, determine `s.<id_prop>` from the schema.
5. Ensure the query uses exact node labels and relationship types as defined in the 'Original Cypher Queries'.
6. For string comparisons on properties (e.g., `s.title = '{subject}'`), consider if case sensitivity is implied by schema or if `toLower()` or `CONTAINS` might be more robust, but default to exact match unless schema suggests otherwise. For this task, prefer exact matches on identifiers.
7. The RETURN clause MUST yield three columns: `subject`, `predicate`, `object` that represent the found fact in the graph.

Example for relationship (if triplet is ('The Matrix', 'HAS_GENRE', 'Sci-Fi') and schema has `(s:Show {{title: row.title}})-[:HAS_GENRE]->(g:Genre {{name: trim(genre_name)}})`):
```cypher
MATCH (s:Show)-[r:HAS_GENRE]->(o:Genre)
WHERE s.title = '{subject}' AND o.name = '{object_}'
RETURN s.title AS subject, type(r) AS predicate, o.name AS object
```

Example for property (if triplet is ('The Matrix', 'HAS_PROPERTY_year', '1999') and schema has `(s:Show {{title: row.title}})` and `SET s.year = toInteger(row.year)`):
```cypher
MATCH (s:Show)
WHERE s.title = '{subject}' AND s.year = toInteger('{object_}')
RETURN s.title AS subject, 'HAS_PROPERTY_year' AS predicate, s.year AS object
```
CRITICAL: Return only the single, complete Cypher query. No additional text, no explanations, no "```cypher" markers.
"""
            llm3_user_prompt = f"""Triplet to query for:
Subject: {subject}
Predicate: {predicate}
Object: {object_}

Sample Data (from CSV, for context only):
{json.dumps(llm_context['data_sample'], indent=2)}

Original Cypher Queries (Knowledge Graph Structure - use this to guide query generation):
```cypher
{llm_context['generated_kg_cypher_schema']}
```
"""
            validation_cypher_query_raw = call_openai_llm(
                llm3_system_prompt,
                llm3_user_prompt,
                openai_client,
                OPENAI_MODEL_NAME,
                OPENAI_GENERATION_CONFIG
            )
            
            premises_found_for_hypothesis = [] # Renamed from premise_triplets_found

            if not validation_cypher_query_raw:
                print(f"  LLM3 failed to generate validation query for {hyp_triplet}.")
                validation_cypher_query = "-- LLM FAILED TO GENERATE VALIDATION QUERY --"
            else:
                validation_cypher_query = validation_cypher_query_raw.strip()
                if validation_cypher_query.startswith("```cypher"):
                    validation_cypher_query = validation_cypher_query[len("```cypher"):].strip()
                if validation_cypher_query.endswith("```"):
                    validation_cypher_query = validation_cypher_query[:-len("```")].strip()
                
                print(f"  LLM3 Generated Validation Query: {validation_cypher_query}")
                query_results = run_cypher_query(neo4j_driver, validation_cypher_query)

                if query_results is not None:
                    for record in query_results:
                        if 'subject' in record and 'predicate' in record and 'object' in record:
                           premises_found_for_hypothesis.append((str(record['subject']), str(record['predicate']), str(record['object'])))
                        else:
                           print(f"Warning: KG validation query result record missing expected keys: {record}")
            
            # Part 2: If hypothesis not found, try to get the actual value from KG
            if not premises_found_for_hypothesis:
                print(f"  Hypothesis {hyp_triplet} NOT directly found. Attempting to find actual KG data...")
                # We need a new LLM call (LLM4 or enhanced LLM3) or rule-based logic here
                # For now, let's design a prompt for a new LLM4 call
                llm4_system_prompt = """You are a Neo4j Cypher query expert. 
Given a hypothesis triplet (Subject, Predicate, Object_hypothesis) that was NOT found in the KG, 
and the KG schema, your task is to generate a Cypher query to retrieve the ACTUAL object(s) or property value(s) 
for the given Subject and Predicate from the Knowledge Graph. 
Focus on retrieving what IS in the KG for that Subject-Predicate pair."""
                llm4_user_prompt = f"""Original Hypothesis (NOT found as stated):
Subject: {subject}
Predicate: {predicate}
Object (Hypothesized): {object_}

Knowledge Graph Schema (Cypher DDL used to create it):
```cypher
{llm_context['generated_kg_cypher_schema']}
```

Task:
Generate a Cypher query to find the actual data in the KG for '{subject}' regarding '{predicate}'.
1. If Predicate is a relationship (e.g., `HAS_GENRE`, `DIRECTED_BY`):
   - Query for `(s)-[:{predicate}]->(o)` where `s` matches the Subject.
   - Return `s.<id_prop> AS subject, type(r) AS predicate, o.<id_prop> AS actual_object` (or `collect(o.<id_prop>)` if multiple objects are expected for that relationship).
   - Determine `<id_prop>` from the KG schema.
2. If Predicate is a property assertion (e.g., `HAS_PROPERTY_year`):
   - Extract property key (e.g., 'year' from `HAS_PROPERTY_year`).
   - Query for `(s)` where `s` matches the Subject.
   - Return `s.<id_prop> AS subject, '{predicate}' AS predicate, s.<property_key> AS actual_object`.
   - Determine `<id_prop>` and `<property_key>` from the KG schema.

Example for property: If hypothesis was ('MovieX', 'HAS_PROPERTY_year', '2000') and it was NOT found,
and schema shows `(m:Movie {{title: ..., year: ...}})`, you might generate:
`MATCH (m:Movie {{title: '{subject}'}})
 RETURN m.title AS subject, 'HAS_PROPERTY_year' AS predicate, m.year AS actual_object`

Example for relationship: If hypothesis was ('MovieX', 'HAS_GENRE', 'Action') and it was NOT found,
and schema shows `(m:Movie)-[r:HAS_GENRE]->(g:Genre)`, you might generate:
`MATCH (m:Movie {{title: '{subject}'}})-[r:HAS_GENRE]->(g:Genre)
 RETURN m.title AS subject, type(r) AS predicate, g.name AS actual_object`
 (Or use COLLECT if multiple genres are expected for a movie based on schema)

Return ONLY the Cypher query. No explanations or markdown.
"""
                correction_query_cypher = call_openai_llm(
                    llm4_system_prompt,
                    llm4_user_prompt,
                    openai_client,
                    OPENAI_MODEL_NAME, # Could use a different model if needed, e.g., a cheaper/faster one for this focused task
                    OPENAI_GENERATION_CONFIG
                )
                current_llm4_correction_query_str = correction_query_cypher # Store the generated query string

                actual_kg_trips_for_hyp = [] # This is the list of (s,p,o) triplets from LLM4's query
                if correction_query_cypher and not correction_query_cypher.startswith("--") : # Check if LLM generated a query
                    correction_query_cypher = correction_query_cypher.strip()
                    if correction_query_cypher.startswith("```cypher"):
                       correction_query_cypher = correction_query_cypher[len("```cypher"):].strip()
                    if correction_query_cypher.endswith("```"):
                       correction_query_cypher = correction_query_cypher[:-len("```")].strip()
                    print(f"  LLM4 Generated Correction Query: {correction_query_cypher}")
                    correction_results = run_cypher_query(neo4j_driver, correction_query_cypher)
                    if correction_results is not None:
                        for rec in correction_results:
                            # The correction query should return subject, predicate, actual_object
                            if 'subject' in rec and 'predicate' in rec and 'actual_object' in rec:
                                actual_kg_trips_for_hyp.append((
                                    str(rec['subject']), 
                                    str(rec['predicate']), 
                                    str(rec['actual_object']))
                                )
                            else:
                                print(f"Warning: Correction query result record missing expected keys: {rec}")
                else:
                    print(f"  LLM4 failed to generate a correction query for {hyp_triplet} or returned no query.")
                
                actual_kg_trips_found_by_llm4 = actual_kg_trips_for_hyp # Store the results for validation_results
                
                # Store the actual KG triplets found for populating all_premise_triplets_final
                actual_kg_triplets_map[hyp_triplet] = actual_kg_trips_for_hyp # This map is used for overall premise gathering
                if actual_kg_trips_for_hyp: 
                    all_premise_triplets_final.extend(actual_kg_trips_for_hyp)
            
            # Populate all_premise_triplets_final if hypothesis was directly supported
            elif premises_found_for_hypothesis: # This is the 'else' to 'if not premises_found_for_hypothesis'
                all_premise_triplets_final.extend(premises_found_for_hypothesis)

            # Run Hallucination Detection
            hallucination_detection_outputs = []
            premise_sentences_for_hf_detection = [] # Sentences to use for HF model

            if premises_found_for_hypothesis: # Prefer direct premises if available
                premise_sentences_for_hf_detection = [format_triplet_as_sentence(prem_trip) for prem_trip in premises_found_for_hypothesis]
                print(f"  Running hallucination detection against {len(premise_sentences_for_hf_detection)} KG premise sentence(s) from direct validation (LLM3)...")
            elif actual_kg_trips_found_by_llm4: # Fallback to LLM4's findings if LLM3 found nothing
                premise_sentences_for_hf_detection = [format_triplet_as_sentence(actual_trip) for actual_trip in actual_kg_trips_found_by_llm4]
                print(f"  Running hallucination detection against {len(premise_sentences_for_hf_detection)} actual KG sentence(s) from correction query (LLM4)...")

            current_best_score = None
            current_overall_hallucinated_status = None # Can be True, False, or None if not determined
            rule_based_corrected_triplet = None # Initialize for this hypothesis
            generative_ai_corrected_sentence = None # Initialize for LLM5 correction

            if hf_hallucination_model and premise_sentences_for_hf_detection and hypothesis_sentence_for_hf:
                hallucination_detection_outputs = run_hallucination_detection_batch(
                    hf_hallucination_model,
                    premise_sentences_for_hf_detection,
                    hypothesis_sentence_for_hf
                )
                if hallucination_detection_outputs:
                    # Determine best score and overall status from these outputs
                    temp_best_score = -1.0 
                    any_non_hallucinated = False
                    valid_scores_found = False
                    for detect_result in hallucination_detection_outputs:
                        if detect_result['consistency_score'] is not None:
                            valid_scores_found = True
                            if detect_result['consistency_score'] > temp_best_score:
                                temp_best_score = detect_result['consistency_score']
                            if detect_result['is_hallucinated'] == False:
                                any_non_hallucinated = True
                    
                    if valid_scores_found:
                        current_best_score = temp_best_score
                        # Option 1: Overall status based on any non-hallucinated premise
                        # current_overall_hallucinated_status = not any_non_hallucinated 
                        # Option 2: Overall status based on best score vs threshold
                        current_overall_hallucinated_status = current_best_score < HALLUCINATION_THRESHOLD
                        
                        # Perform rule-based correction if hallucination is detected and premises are available
                        if current_overall_hallucinated_status is True:
                            # Determine which premises to use for rule-based correction
                            # Default to actual_kg_trips_found_by_llm4 if LLM3 found nothing but LLM4 did
                            # Prefer premises_found_for_hypothesis if LLM3 found something directly
                            candidate_premises_for_correction = []
                            if premises_found_for_hypothesis:
                                candidate_premises_for_correction = premises_found_for_hypothesis
                            elif actual_kg_trips_found_by_llm4:
                                candidate_premises_for_correction = actual_kg_trips_found_by_llm4
                            
                            if candidate_premises_for_correction:
                                # Use the first premise triplet for rule-based correction
                                first_premise_for_correction = candidate_premises_for_correction[0]
                                premise_subject, premise_predicate, premise_object = first_premise_for_correction
                                
                                # If premise_object is a list (e.g. from Cypher collect()), take the first element
                                corrected_object = premise_object
                                if isinstance(premise_object, list) and premise_object:
                                    corrected_object = premise_object[0]
                                elif isinstance(premise_object, list) and not premise_object: # empty list
                                    corrected_object = "[No value found in KG list]" # Or some other placeholder

                                rule_based_corrected_triplet = (hyp_triplet[0], hyp_triplet[1], corrected_object)

                            # Now, attempt LLM5 generative correction
                            if openai_client: # Check if client is available
                                llm5_system_prompt = """You are an expert in editing sentences to correct factual inaccuracies based on provided knowledge.
You will receive an original sentence, a specific factual claim (triplet) made in that sentence that has been identified as a hallucination, and a corresponding correct factual triplet from a knowledge graph.
Your task is to rewrite the original sentence so that the part reflecting the hallucinated triplet is corrected according to the knowledge graph fact.
Output only the single, complete, corrected sentence. Maintain the original tone and style as much as possible.
If the knowledge graph provides a list as the correct object (e.g., for multiple genres like '[Drama, Thriller]'), try to incorporate it naturally (e.g., 'is a Drama and Thriller movie') or pick the most prominent one if a list doesn't fit well fluently into the original sentence structure. If the list contains only one item, use that item directly.
Avoid starting with phrases like "Corrected sentence:"."""
                                
                                # Prepare premise for LLM5: use the corrected_object for consistency
                                premise_for_llm5 = (premise_subject, premise_predicate, corrected_object) # Using the potentially simplified object

                                llm5_user_prompt = f"""Original User Sentence:
{user_sentence}

Hallucinated Triplet (extracted from the original sentence):
Subject: {hyp_triplet[0]}
Predicate: {hyp_triplet[1]}
Object: {hyp_triplet[2]}

Correct Fact from Knowledge Graph (use this to correct the sentence):
Subject: {premise_for_llm5[0]}
Predicate: {premise_for_llm5[1]}
Object: {premise_for_llm5[2]}

Please provide the corrected full sentence based on the knowledge graph fact, ensuring it remains a single coherent sentence.
"""
                                print(f"    Calling LLM5 for generative correction of hypothesis: {hyp_triplet}...")
                                generative_ai_corrected_sentence = call_openai_llm(
                                    llm5_system_prompt,
                                    llm5_user_prompt,
                                    openai_client,
                                    OPENAI_MODEL_NAME, # Or a model specialized for editing if available/preferred
                                    OPENAI_GENERATION_CONFIG 
                                )
                                if generative_ai_corrected_sentence:
                                    print(f"      LLM5 Corrected Sentence: {generative_ai_corrected_sentence}")
                                else:
                                    print("      LLM5 failed to generate a corrected sentence.")

                    else: # Errors or no valid scores in detection_outputs
                        current_overall_hallucinated_status = None # Undetermined
                        current_best_score = None 
                else:
                    print("    Hallucination detection ran but did not return structured results.")
                    current_overall_hallucinated_status = None # Undetermined
                    current_best_score = None
            elif not hf_hallucination_model:
                print("    Skipping hallucination detection (Hugging Face model not loaded).")
            elif not premise_sentences_for_hf_detection:
                 print("    Skipping hallucination detection (no KG premise/actual sentences to compare against).")
            elif not hypothesis_sentence_for_hf:
                 print("    Skipping hallucination detection (hypothesis sentence not formatted).")

            validation_results.append({
                "hypothesis_triplet": hyp_triplet,
                "hypothesis_sentence_for_hf": hypothesis_sentence_for_hf,
                "validation_query_llm3": validation_cypher_query,
                "premises_directly_found_llm3": premises_found_for_hypothesis,
                "correction_query_llm4": current_llm4_correction_query_str, 
                "actual_kg_triplets_llm4": actual_kg_trips_found_by_llm4, 
                "hallucination_detection_outputs": hallucination_detection_outputs, # Keep for potential detailed debugging if needed
                "best_consistency_score": current_best_score,
                "overall_hallucination_status": current_overall_hallucinated_status,
                "rule_based_corrected_triplet": rule_based_corrected_triplet,
                "generative_ai_corrected_sentence": generative_ai_corrected_sentence
            })

            # If a hallucination was detected, store it for the final correction step
            if current_overall_hallucinated_status is True and rule_based_corrected_triplet:
                all_hallucinated_triplets.append(hyp_triplet)
                all_correction_triplets.append(rule_based_corrected_triplet)

            time.sleep(1) 

    # 7. LLM Call 5 (Single, Final Call): Generative Sentence Correction
    final_generative_ai_corrected_sentence = None
    if all_hallucinated_triplets and openai_client:
        print("\n--- Step 6: Generating Final Corrected Sentence (LLM5) ---")
        llm5_system_prompt = """You are an expert in editing sentences to correct factual inaccuracies based on provided knowledge.
You will receive an original sentence, a list of factual claims (triplets) from that sentence identified as hallucinations, and a corresponding list of correct factual triplets from a knowledge graph.
Your task is to rewrite the original sentence in a single, coherent way that corrects *all* identified inaccuracies using the provided knowledge graph facts.
Maintain the original tone and style as much as possible.
If the knowledge graph provides a list as the correct object (e.g., for multiple genres like '[Drama, Thriller]'), incorporate it naturally (e.g., 'is a Drama and Thriller movie').
Avoid starting with phrases like "Corrected sentence:"."""

        # Format the lists of triplets for the prompt
        hallucinated_triplets_str = "\n".join([f"- {trip}" for trip in all_hallucinated_triplets])
        correct_triplets_str = "\n".join([f"- {trip}" for trip in all_correction_triplets])

        llm5_user_prompt = f"""Original User Sentence:
{user_sentence}

Hallucinated Triplets (extracted from the original sentence):
{hallucinated_triplets_str}

Correct Facts from Knowledge Graph (use these to correct the sentence):
{correct_triplets_str}

Please provide the single, comprehensively corrected sentence based on all the knowledge graph facts.
"""
        print(f"    Calling LLM5 for final generative correction...")
        final_generative_ai_corrected_sentence = call_openai_llm(
            llm5_system_prompt,
            llm5_user_prompt,
            openai_client,
            OPENAI_MODEL_NAME,
            OPENAI_GENERATION_CONFIG
        )
        if final_generative_ai_corrected_sentence:
            print(f"      LLM5 Final Corrected Sentence: {final_generative_ai_corrected_sentence}")
        else:
            print("      LLM5 failed to generate a final corrected sentence.")

    # 8. Output Results
    print("\n--- Hallucination Review and Correction Summary ---")
    if not validation_results:
        print("No validation was performed (likely no hypothesis triplets were extracted).")
    else:
        for item in validation_results:
            print(f"\n--------------------------------------------------")
            print(f"Hypothesis Triplet: {item['hypothesis_triplet']}")

            kg_premises_for_display = []
            if item['premises_directly_found_llm3']:
                kg_premises_for_display.extend(item['premises_directly_found_llm3'])
            elif item['actual_kg_triplets_llm4']:
                kg_premises_for_display.extend(item['actual_kg_triplets_llm4'])
            
            if kg_premises_for_display:
                print("  KG Premise(s) Used for Validation:")
                for prem in kg_premises_for_display:
                    if isinstance(prem, tuple) and len(prem) == 3 and isinstance(prem[2], list):
                         # Handle cases where premise object itself is a list (e.g. from collect() in Cypher)
                         obj_display = f"{prem[2]}" # Show the full list in this display part
                         print(f"    - ({prem[0]}, {prem[1]}, {obj_display})")
                    else:
                         print(f"    - {prem}")
            else:
                print("  No direct or alternative premises found in KG for comparison.")

            score_display = f"{item['best_consistency_score']:.4f}" if item['best_consistency_score'] is not None else "N/A"
            status_display = "Undetermined"
            if item['overall_hallucination_status'] is True:
                status_display = "HALLUCINATION"
            elif item['overall_hallucination_status'] is False:
                status_display = "Likely NOT a Hallucination"
            
            print(f"  Best Consistency Score: {score_display}")
            print(f"  Status: {status_display}")

            if item['overall_hallucination_status'] is True and item['rule_based_corrected_triplet']:
                print(f"  Rule-Based Corrected Triplet: {item['rule_based_corrected_triplet']}")
            
            print(f"--------------------------------------------------")

    # Display the final, single corrected sentence
    if final_generative_ai_corrected_sentence:
        print("\n--- Final Corrected Sentence ---")
        print(final_generative_ai_corrected_sentence)
    elif all_hallucinated_triplets:
        print("\n--- Final Corrected Sentence ---")
        print("Could not generate a final corrected sentence, but hallucinations were detected.")
    else:
        print("\n--- Final Sentence Review ---")
        print("No hallucinations were detected in the original sentence.")

    # Cleanup
    if neo4j_driver:
        neo4j_driver.close()
        print("\nNeo4j connection closed.")

if __name__ == "__main__":
    main_pipeline()
    print("\nPipeline finished.") 