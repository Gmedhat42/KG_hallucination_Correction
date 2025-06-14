import streamlit as st

# Configure Streamlit theme for light mode
st.set_page_config(
    page_title="Using Knowledge Graphs for Hallucination Correction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set light theme
st.markdown("""
    <style>
        /* Light mode styles */
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        
        /* Make text darker for better contrast in light mode */
        .stMarkdown, .stText {
            color: #000000;
        }
        
        /* Style buttons in light mode */
        .stButton button {
            background-color: #f0f2f6;
            color: #000000;
            border: 1px solid #e0e3e9;
        }
        
        /* Style sidebar in light mode */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Style expander headers in light mode */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            color: #000000;
        }
        
        /* Style dataframe in light mode */
        .stDataFrame {
            background-color: #ffffff;
        }
        
        /* Style tabs in light mode */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8f9fa;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #000000;
        }
        
        /* Style code blocks in light mode */
        .stCodeBlock {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

import tempfile
import os
import builtins
import io
import re
import pandas as pd
from contextlib import redirect_stdout
from openai import OpenAI

# External
from neo4j import GraphDatabase  # Used for connection verification

# Import the original script so we can reuse its functions without modification
import general_pipeline.general_pipeline as general_pipeline  # noqa: E402


def run_old_pipeline(
    csv_path: str,
    clear_db: bool,
    user_sentence: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> str:
    """Runs old_generic.main_pipeline with patched `input()` values.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file to ingest.
    clear_db : bool
        Whether to clear the Neo4j database before loading new data.
    user_sentence : str
        Sentence provided by the user for hallucination checking.
    neo4j_uri : str
        URI of the Neo4j database.
    neo4j_user : str
        Username for the Neo4j database.
    neo4j_password : str
        Password for the Neo4j database.

    Returns
    -------
    str
        Captured stdout log produced by the original pipeline.
    """

    # Update the global connection details of old_generic before running
    general_pipeline.NEO4J_URI = neo4j_uri
    general_pipeline.NEO4J_USERNAME = neo4j_user
    general_pipeline.NEO4J_PASSWORD = neo4j_password

    # Prepare predetermined answers for every `input()` call inside main_pipeline
    # 1) CSV path, 2) clear database?, 3) user sentence.
    # For any additional unexpected prompts we answer "yes" by default so that the
    # pipeline can proceed non-interactively.
    responses = [csv_path, "yes" if clear_db else "no", user_sentence]

    def patched_input(_: str = "") -> str:  # noqa: D401
        return responses.pop(0) if responses else "yes"

    # Capture stdout so we can show it nicely in Streamlit
    buffer = io.StringIO()
    original_input = builtins.input
    builtins.input = patched_input
    try:
        with redirect_stdout(buffer):
            general_pipeline.main_pipeline()
    finally:
        builtins.input = original_input  # Ensure we always restore input

    return buffer.getvalue()


def verify_neo4j_connection(uri: str, user: str, pwd: str) -> tuple[bool, str]:
    """Attempt to connect to Neo4j and return (success, message)."""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        driver.close()
        return True, "Connection successful!"
    except Exception as exc:  # noqa: BLE001
        return False, f"Connection failed: {exc}"


def parse_pipeline_logs(raw: str) -> dict:
    """Extract structured info from the pipeline stdout."""
    lines = raw.splitlines()

    # Extract final corrected sentence
    final_sentence: str | None = None
    for marker in ("LLM5 Final Corrected Sentence:", "--- Final Corrected Sentence ---"):
        if marker in raw:
            segment = raw.split(marker, 1)[1].strip()
            first_line = segment.splitlines()[0] if segment else ""
            final_sentence = first_line.strip().strip("\r\n")
            break

    # Extract triplets list
    triplets: list[str] = []
    capture = False
    for ln in lines:
        if "Extracted" in ln and "hypothesis triplets" in ln:
            capture = True
            continue
        if capture:
            if ln.strip() == "":
                break
            if "- (" in ln:
                triplets.append(ln.strip(" -"))

    # Per-triplet analysis: capture blocks starting with "Validating hypothesis triplet"
    analyses: dict[str, str] = {}
    current_triplet_key = None
    current_block: list[str] = []
    for ln in lines:
        if ln.startswith("Validating hypothesis triplet"):
            # flush previous
            if current_triplet_key is not None:
                analyses[current_triplet_key] = "\n".join(current_block)
            # start new
            current_triplet_key = ln.split(":", 1)[1].strip()
            current_block = [ln]
        elif current_triplet_key is not None:
            if ln.startswith("Validating hypothesis triplet"):
                # should be handled by if
                pass
            else:
                current_block.append(ln)
    # flush last
    if current_triplet_key is not None:
        analyses[current_triplet_key] = "\n".join(current_block)

    # Parse concise hallucination summary
    summary_items: list[dict] = []
    try:
        summary_start = raw.index("--- Hallucination Review and Correction Summary ---")
        summary_section = raw[summary_start:].split("--- Final", 1)[0]
        for block in summary_section.split("Hypothesis Triplet:")[1:]:
            header_line, *rest = block.strip().splitlines()
            triplet_str = header_line.strip()
            status_line = next((l for l in rest if l.strip().startswith("Status:")), None)
            corrected_line = next((l for l in rest if l.strip().startswith("Rule-Based Corrected Triplet:")), None)
            status = status_line.split(":", 1)[1].strip() if status_line else "Undetermined"
            corrected_triplet = corrected_line.split(":", 1)[1].strip() if corrected_line else None
            summary_items.append({
                "triplet": triplet_str,
                "status": status,
                "corrected": corrected_triplet,
            })
    except Exception:
        pass

    return {
        "triplets": triplets,
        "analyses": analyses,
        "final_sentence": final_sentence,
        "summary": summary_items,
    }


def generate_sample_claims(df: pd.DataFrame) -> list[str]:
    """Generate sample claims about the dataset using GPT-4.
    
    Args:
        df: DataFrame containing the dataset
    
    Returns:
        List of generated claims
    """
    try:
        client = OpenAI(api_key=general_pipeline.OPENAI_API_KEY)
        
        # Prepare sample data for the prompt
        sample_data = df.head(5).to_dict(orient='records')
        columns_info = {col: df[col].dtype for col in df.columns}
        
        system_prompt = """You are an expert in generating realistic but potentially incorrect claims about a dataset.
        Your task is to generate claims that can be verified against our specific dataset knowledge graph.
        
        Dataset Information:
        Columns and Types:
        {columns}
        
        Sample Data:
        {sample}
        
        Generate 5 claims that:
        1. Are realistic and sound plausible
        2. Can be verified against our knowledge graph
        3. May contain common misconceptions or incorrect information
        4. Use natural, conversational language
        5. Are concise (1-2 sentences each)
        6. Focus on relationships and facts that can be verified (e.g., directors, actors, ratings, years, genres)
        
        Your examples should easy to test
         
        Format each claim on a new line, starting with a number and period.
        Return ONLY the numbered claims, no additional text."""

        formatted_prompt = system_prompt.format(
            columns="\n".join(f"- {col}: {dtype}" for col, dtype in columns_info.items()),
            sample="\n".join(str(record) for record in sample_data)
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": "Please generate 5 realistic but potentially incorrect claims about our dataset."}
            ],
            temperature=0.7
        )
        
        # Extract claims from response
        claims_text = response.choices[0].message.content.strip()
        # Split by newline and number pattern (e.g., "1.", "2.")
        claims = [claim.strip() for claim in re.split(r'\d+\.', claims_text) if claim.strip()]
        
        return claims
    except Exception as e:
        st.error(f"Error generating claims: {e}")
        return []


# ============================= Streamlit UI ============================= #

# Initialize session state for progress tracking
if "step" not in st.session_state:
    st.session_state.step = 1  # 1: Connect, 2: Upload, 3: Build, 4: Analyze
if "kg_built" not in st.session_state:
    st.session_state.kg_built = False
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ---------------------------------------------------------------------------------
# 🔗 Sidebar – Neo4j Connection & Progress
# ---------------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 KG Inspector")
    
    # Progress tracker
    st.markdown("### Progress")
    steps = [
        "1. Connect DB 🔌",
        "2. Load Data 📊",
        "3. Build Graph 🏗️",
        "4. Analyze ✨"
    ]
    for i, step in enumerate(steps, 1):
        if i < st.session_state.step:
            st.success(step)
        elif i == st.session_state.step:
            st.info(step)
        else:
            st.markdown(f"_{step}_")
    
    st.divider()
    
    # Connection settings
    with st.expander("🔗 Neo4j Connection", expanded=True):
        neo4j_uri = st.text_input(
            "URI",
            value="bolt://localhost:7687",
            key="uri",
            help="The URI of your Neo4j database instance"
        )
        neo4j_user = st.text_input(
            "Username",
            value="neo4j",
            key="user",
            help="Your Neo4j username"
        )
        neo4j_password = st.text_input(
            "Password",
            type="password",
            key="pwd",
            help="Your Neo4j password"
        )

        if st.button("🔄 Test Connection", key="verify_conn", use_container_width=True):
            with st.spinner("Testing connection..."):
                ok, msg = verify_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
                if ok:
                    st.success("✅ " + msg)
                    st.session_state.step = max(2, st.session_state.step)
                else:
                    st.error("❌ " + msg)

    # Store credentials
    st.session_state.update({
        "neo4j_uri": neo4j_uri,
        "neo4j_user": neo4j_user,
        "neo4j_password": neo4j_password
    })

    # Reset button at bottom
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------------

st.title("Using Knowledge Graphs for Hallucination Correction")

with st.expander("ℹ️ About this tool", expanded=not st.session_state.get("kg_built")):
    st.markdown(
        """
        This interactive tool helps you:
        1. **Build** a Neo4j knowledge graph from any CSV dataset
        2. **Validate** natural language statements against that graph
        3. **Detect & Correct** hallucinations in those statements
        
        Perfect for fact-checking, data validation, and exploring how your data can
        catch potential misinformation. Get started by connecting to Neo4j and uploading
        your CSV!
        """
    )

# ---------------------------------------------------------------------------------
# 2️⃣ CSV Upload & Preview
# ---------------------------------------------------------------------------------

st.header("Dataset Selection")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="csv_up",
        help="Select a CSV file to build your knowledge graph from"
    )
with col2:
    custom_path = st.text_input(
        "…or enter existing CSV path",
        key="csv_path",
        help="Alternatively, provide the path to an existing CSV file"
    )

csv_path: str | None = None
if uploaded_file is not None:
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_path = tmp_path
    st.session_state.step = max(3, st.session_state.step)
elif custom_path:
    csv_path = custom_path.strip()
    if os.path.isfile(csv_path):
        st.session_state.step = max(3, st.session_state.step)

# Show data preview if available
if csv_path and os.path.isfile(csv_path):
    try:
        df = pd.read_csv(csv_path)
        st.session_state.data_loaded = True
        
        # Preview tabs
        tab1, tab2, tab3 = st.tabs(["📊 Preview", "📋 Schema", "🎯 Sample Claims"])
        
        with tab1:
            st.dataframe(
                df.head(1000),
                use_container_width=True,
                height=300
            )
            if len(df) > 1000:
                st.caption(f"Showing first 1,000 of {len(df):,} rows")
                
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Column Types")
                for col in df.columns:
                    st.code(f"{col}: {df[col].dtype}")
            with col2:
                st.markdown("##### Sample Values")
                for col in df.columns:
                    st.markdown(f"**{col}**: {', '.join(map(str, df[col].dropna().unique()[:3]))}")
        
        with tab3:
            st.markdown("### 🎯 Sample Claims for Testing")
            st.markdown("""
            Below are some automatically generated claims about your dataset. 
            These claims may contain inaccuracies and are perfect for testing the hallucination detection system.
            Click the copy button next to any claim to use it in the analysis.
            """)
            
            # Generate claims button at the top
            if st.button("🎲 Generate Claims", type="primary", use_container_width=True):
                with st.spinner("Generating claims..."):
                    st.session_state.sample_claims = generate_sample_claims(df)
                st.rerun()  # Use st.rerun() instead of experimental_rerun
            
            # Display claims if they exist
            if "sample_claims" in st.session_state and st.session_state.sample_claims:
                for i, claim in enumerate(st.session_state.sample_claims, 1):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{i}.** {claim}")
                    with col2:
                        if st.button("📋 Copy", key=f"copy_claim_{i}"):
                            st.session_state.sentence = claim
                            st.rerun()  # Use st.rerun() instead of experimental_rerun
                    st.divider()
            else:
                st.info("Click 'Generate Claims' to create sample claims for testing.")
                    
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.session_state.data_loaded = False

# ---------------------------------------------------------------------------------
# 3️⃣ Knowledge Graph Construction
# ---------------------------------------------------------------------------------

if st.session_state.data_loaded:
    st.header("Knowledge Graph Construction")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        clear_db = st.checkbox(
            "Clear existing Neo4j database before loading",
            value=True,
            key="clear_db",
            help="Check to start fresh, uncheck to add to existing graph"
        )
    with col2:
        build_btn = st.button(
            "🏗️ Build Knowledge Graph",
            type="primary",
            disabled=not csv_path,
            use_container_width=True
        )

    if build_btn:
        progress_text = "Building knowledge graph..."
        my_bar = st.progress(0, text=progress_text)

        try:
            # Provide blank sentence so pipeline stops after KG build
            with st.spinner(progress_text):
                my_bar.progress(25, text="Initializing...")
                logs = run_old_pipeline(
                    csv_path,
                    clear_db,
                    "",  # blank sentence
                    st.session_state["neo4j_uri"],
                    st.session_state["neo4j_user"],
                    st.session_state["neo4j_password"],
                )
                my_bar.progress(75, text="Loading data...")
                st.session_state.kg_built = True
                st.session_state.kg_logs = logs
                my_bar.progress(100, text="Complete!")
                st.success("✨ Knowledge graph construction completed!")
                st.session_state.step = max(4, st.session_state.step)
                
        except Exception as e:
            st.error("Failed to build knowledge graph")
            st.exception(e)
            st.session_state.kg_built = False
        finally:
            my_bar.empty()

# ---------------------------------------------------------------------------------
# 4️⃣ Sentence Analysis
# ---------------------------------------------------------------------------------

if st.session_state.kg_built:
    st.header("Sentence Analysis")

    sentence = st.text_area(
        "Enter a sentence to validate",
        key="sentence",
        help="Write a statement about your data that you want to fact-check",
        placeholder="Example: 'Forrest Gump was directed by Steven Spielberg and had an IMDb rating of 9.5'"
    )

    analyze_btn = st.button(
        "🔍 Analyze Statement",
        type="primary",
        disabled=not sentence.strip(),
        use_container_width=True
    )

    if analyze_btn:
        with st.spinner("Analyzing your statement..."):
            try:
                analysis_logs = run_old_pipeline(
                    csv_path,
                    False,  # do NOT clear DB again
                    sentence,
                    st.session_state["neo4j_uri"],
                    st.session_state["neo4j_user"],
                    st.session_state["neo4j_password"],
                )
            except Exception as e:
                st.error("Analysis failed")
                st.exception(e)
                st.stop()

        parsed = parse_pipeline_logs(analysis_logs)

        # Results in tabs
        tab1, tab2, tab3 = st.tabs(["✨ Summary", "🔍 Details", "📝 Full Log"])

        with tab1:
            if parsed["final_sentence"]:
                st.markdown("### 🎯 Corrected Sentence")
                st.markdown(
                    f"""
                    <div style='padding: 20px; border-radius: 5px; background-color: #e6f3ff; border-left: 5px solid #0066cc;'>
                    <p style='margin:0; font-size: 1.1em;'>{parsed["final_sentence"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.divider()
            
            if parsed["triplets"]:
                st.markdown("##### Extracted Claims")
                for trip in parsed["triplets"]:
                    st.code(trip, language="text")

        with tab2:
            if parsed.get("summary"):
                for item in parsed["summary"]:
                    trip = item["triplet"]
                    status = item["status"]
                    corrected = item.get("corrected")

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        color = (
                            "green" if status == "Likely NOT a Hallucination"
                            else "orange" if status == "Undetermined"
                            else "red"
                        )
                        st.markdown(
                            f"<div style='padding:10px;border-left:3px solid {color}'>"
                            f"<code>{trip}</code><br>"
                            f"<small style='color:{color}'><b>{status}</b></small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        if status == "HALLUCINATION" and corrected:
                            st.markdown(f"**Correct value:**\n`{corrected}`")

        with tab3:
            st.text(analysis_logs)
