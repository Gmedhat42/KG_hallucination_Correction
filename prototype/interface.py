# File: app.py (Enhanced with Pipeline Integration and Improved UX)

import streamlit as st
import pandas as pd
import os
import re
import json
import time
from prototype.main_pipeline import run_correction_pipeline, generate_hallucinated_claims
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph

# --- Page Configuration (Must be first Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="KG Hallucination Detection & Correction",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
CLEANED_CSV_PATH = "drug_data.csv"

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm_generator = None
neo4j_graph = None

if not OPENAI_API_KEY:
    st.warning("Warning: OPENAI_API_KEY not found.")
else:
    try:
        llm_generator = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.2)
    except Exception as e:
        st.error(f"Error initializing OpenAI LLM: {e}")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    st.warning("Warning: Neo4j connection details missing or incomplete.")
else:
    try:
        neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        neo4j_graph.query("RETURN 1")
    except Exception as e:
        st.error(f"Error connecting to Neo4j: {e}")

# --- Custom CSS for better UX ---
st.markdown("""
<style>
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .processing-text {
        font-size: 1.2em;
        color: #666;
        font-style: italic;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #434343;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .correction-box {
        background-color: #434343;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .stTextArea textarea {
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 1px #4CAF50 !important;
    }
    .processing-textbox textarea {
        background-color: #f8f9fa !important;
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 1px #4CAF50 !important;
    }
    .success-textbox textarea {
        background-color: #e8f5e9 !important;
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 1px #4CAF50 !important;
    }
    .error-textbox textarea {
        background-color: #ffebee !important;
        border-color: #f44336 !important;
        box-shadow: 0 0 0 1px #f44336 !important;
    }
</style>
""", unsafe_allow_html=True)

def generate_correction(original_text: str, correct_triples: list, llm_client: ChatOpenAI) -> str:
    """
    Uses generative AI to create a natural correction based on the correct triples.
    """
    if not correct_triples:
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

# --- Load Data ---
@st.cache_data
def load_data(csv_path):
    """Loads the cleaned drug data CSV."""
    if not os.path.exists(csv_path):
        st.error(f"Error: Cleaned data file not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path).fillna('')
        print(f"Data loaded successfully from {csv_path}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Helper Function for Multi-Column Search ---
def search_dataframe(df, search_terms):
    """
    Filters DataFrame based on search terms entered for specific columns.
    Search is case-insensitive and checks if term is contained within the cell value.
    """
    df_filtered = df.copy()
    for column, term in search_terms.items():
        if term and column in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(term, case=False, na=False, regex=True)]
    return df_filtered

# --- Sidebar ---
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application demonstrates how Knowledge Graphs can verify and correct factual claims in text.
    
    **Features:**
    - Fact extraction from text
    - Knowledge Graph verification
    - Hallucination detection
    - Factual correction (Rule-based & Generative)
    
    **Example Claims:**
    - "Doxycycline treats Acne and belongs to the Penicillin class"
    - "Spironolactone causes sleepiness"
    - "Accutane is safe for pregnancy"
    """)

# --- Main Content ---
st.title("🧪 KG Hallucination Detection & Correction Demo")

# --- Data Explorer Section ---
st.header("1. Explore the Knowledge Base")
st.markdown("""
Search and explore the drug database to understand the available information.
Use the filters below to find specific drugs, conditions, or drug classes.
""")

# Load the data
data_df_full = load_data(CLEANED_CSV_PATH)

if data_df_full is not None:
    st.write(f"Loaded {len(data_df_full)} drug records.")

    st.subheader("Search & Filter Data")
    st.write("Enter search terms for specific columns (case-insensitive). Leave blank to ignore a column.")

    # Create search boxes in columns
    search_cols = st.columns(4)
    search_terms = {}

    with search_cols[0]:
        search_terms['drug_name'] = st.text_input("Search Drug Name", key="search_drug")
    with search_cols[1]:
        search_terms['medical_condition'] = st.text_input("Search Condition", key="search_cond")
    with search_cols[2]:
        search_terms['generic_name'] = st.text_input("Search Generic Name", key="search_gen")
    with search_cols[3]:
        search_terms['drug_classes'] = st.text_input("Search Drug Class", key="search_class")

    # Apply search/filter
    filtered_df = search_dataframe(data_df_full, search_terms)

    # Display the filtered DataFrame with better styling
    st.dataframe(
        filtered_df,
        height=300,
        use_container_width=True,
        hide_index=True
    )
    st.write(f"Showing {len(filtered_df)} matching records.")

    st.markdown("---")

    # Data Insights Section
    st.subheader("Quick Data Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        unique_drugs = data_df_full['drug_name'].nunique()
        st.metric("Unique Drug Names", unique_drugs)
        if 'generic_name' in data_df_full.columns:
            unique_generic = data_df_full['generic_name'].nunique()
            st.metric("Unique Generic Names", unique_generic)

    with col2:
        if 'medical_condition' in data_df_full.columns:
            unique_conditions = data_df_full['medical_condition'].nunique()
            st.metric("Unique Conditions", unique_conditions)
        if 'drug_classes' in data_df_full.columns:
            all_classes = set()
            try:
                data_df_full['drug_classes'].dropna().astype(str).str.split('|').apply(
                    lambda x: [all_classes.add(c.strip()) for c in x if isinstance(x, list) and c.strip()]
                )
            except AttributeError:
                pass
            st.metric("Unique Drug Classes", len(all_classes))

    with col3:
        if 'rx_otc' in data_df_full.columns:
            st.write("Prescription Status:")
            rx_counts = data_df_full['rx_otc'].value_counts()
            st.bar_chart(rx_counts)

else:
    st.warning("Could not load data. Data explorer is unavailable.")

# --- Pipeline Section (Moved to bottom) ---
st.markdown("---")
st.header("2. Verify and Correct Claims")

# Add Example Claims Section
st.subheader("Example Claims")
st.markdown("""
Generate example claims that can be verified against our knowledge graph.
These claims may contain common misconceptions or incorrect information about drugs.
""")

if st.button("Generate Example Claims"):
    if llm_generator is None or neo4j_graph is None:
        st.error("Cannot generate claims: LLM or Neo4j connection not available.")
    else:
        with st.spinner("Generating example claims..."):
            example_claims = generate_hallucinated_claims(llm_generator, neo4j_graph)
            st.session_state.example_claims = example_claims

if hasattr(st.session_state, 'example_claims'):
    st.markdown("Select a claim to verify:")
    for i, claim in enumerate(st.session_state.example_claims):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{i+1}.** {claim}")
        with col2:
            if st.button("Copy", key=f"copy_{i}"):
                st.session_state.claim_input = claim
                st.rerun()

    st.markdown("---")

st.markdown("""
Enter a claim about a drug to verify its accuracy and correct any hallucinations.
The system will:
1. Extract factual claims
2. Verify them against the knowledge graph
3. Detect any hallucinations
4. Provide corrections using both rule-based and generative approaches
""")

# Create two columns for input and results
input_col, results_col = st.columns([1, 1])

with input_col:
    # Initialize session state for textbox status if not exists
    if 'textbox_status' not in st.session_state:
        st.session_state.textbox_status = 'normal'
    
    # Create a container for the textbox
    textbox_container = st.container()
    
    # Add the textbox with dynamic styling
    with textbox_container:
        user_input = st.text_area(
            "Enter your claim:",
            value=st.session_state.get('claim_input', ''),
            placeholder="Example: Doxycycline treats Acne and belongs to the Penicillin class.",
            height=100,
            key="claim_input"
        )
    
    # Add the verify button
    if st.button("Verify and Correct", type="primary"):
        if user_input:
            # Update textbox status to processing
            st.session_state.textbox_status = 'processing'
            
            # Create a placeholder for the progress bar
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Show initial progress
            progress_bar = progress_placeholder.progress(0)
            status_placeholder.markdown('<p class="processing-text">Initializing verification process...</p>', unsafe_allow_html=True)
            
            # Simulate progress for better UX
            for i in range(4):
                time.sleep(0.5)  # Add small delay for visual feedback
                progress_bar.progress((i + 1) * 25)
                status_messages = [
                    "Extracting facts from your claim...",
                    "Verifying facts against knowledge graph...",
                    "Detecting potential hallucinations...",
                    "Generating corrections..."
                ]
                status_placeholder.markdown(f'<p class="processing-text">{status_messages[i]}</p>', unsafe_allow_html=True)
            
            # Run the actual pipeline
            with st.spinner("Finalizing results..."):
                results = run_correction_pipeline(user_input)
                st.session_state.pipeline_results = results
                
                # Generate AI correction if we have correct triples
                if results['final_triples']:
                    llm_client = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4", temperature=0)
                    ai_correction = generate_correction(user_input, results['final_triples'], llm_client)
                    st.session_state.ai_correction = ai_correction
                
                # Update textbox status based on results
                if results['counts']['hallucinated'] > 0:
                    st.session_state.textbox_status = 'error'
                else:
                    st.session_state.textbox_status = 'success'
            
            # Clear progress indicators
            progress_placeholder.empty()
            status_placeholder.empty()
            
            # Rerun to update the textbox styling
            st.rerun()
        else:
            st.warning("Please enter a claim to verify.")

# Display results if available
if hasattr(st.session_state, 'pipeline_results'):
    results = st.session_state.pipeline_results
    
    with results_col:
        st.subheader("Results")
        
        # Original Text
        st.markdown("**Original Claim:**")
        st.markdown('<div class="info-box">' + results['original_text'] + '</div>', unsafe_allow_html=True)
        
        # Show expanded text if different from original
        if results.get('expanded_text') and results['expanded_text'] != results['original_text']:
            st.markdown("**Expanded Query:**")
            st.markdown('<div class="info-box" style="background-color: #2b2b2b;">' + results['expanded_text'] + '</div>', unsafe_allow_html=True)
            st.info("The query was expanded to better understand the claim. The original meaning is preserved.")
        
        # Extracted Triples
        st.markdown("**Extracted Facts:**")
        if results['extracted_triples']:
            for triple in results['extracted_triples']:
                st.markdown(f'<div class="result-box">- {triple[0]} {triple[1]} {triple[2]}</div>', unsafe_allow_html=True)
        else:
            st.warning("No facts could be extracted.")
        
        # Verification Details
        st.markdown("**Verification Details:**")
        for detail in results['verification_details']:
            if detail.get('is_hallucinated'):
                st.error(f"Hallucination detected: {detail['hypothesis']}")
            else:
                st.success(f"Verified: {detail['hypothesis']}")
        
        # Rule-based Correction
        st.markdown("**Rule-based Correction:**")
        if results['rule_based_text']:
            st.markdown('<div class="correction-box">' + results['rule_based_text'] + '</div>', unsafe_allow_html=True)
        else:
            st.warning("No rule-based correction could be generated.")
        
        # AI-based Correction
        st.markdown("**AI-generated Correction:**")
        if results['generative_text']:
            st.markdown('<div class="correction-box">' + results['generative_text'] + '</div>', unsafe_allow_html=True)
        else:
            st.warning("No AI correction could be generated.")
        
        # Hybrid Correction
        st.markdown("**Hybrid Correction (Combined Approach):**")
        if results['hybrid_text']:
            st.markdown('<div class="correction-box" style="border-left: 4px solid #2196F3;">' + results['hybrid_text'] + '</div>', unsafe_allow_html=True)
            st.info("This correction combines the precision of rule-based correction with the natural flow of AI generation.")
        else:
            st.warning("No hybrid correction could be generated.")
        
        # Statistics
        st.markdown("**Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Facts Extracted", results['counts']['extracted'])
        with col2:
            st.metric("Hallucinations", results['counts']['hallucinated'])
        with col3:
            st.metric("Corrections Made", results['counts']['corrected'])

# --- Footer ---
st.markdown("---")
st.markdown("""
### About This Demo
This application demonstrates the use of Knowledge Graphs for detecting and correcting hallucinations in text.
It uses a combination of:
- LLM-based fact extraction
- Knowledge Graph verification
- Hallucination detection
- Factual correction (Rule-based & Generative)
""") 