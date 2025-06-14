# Knowledge Graph-based Hallucination Detection and Correction

This repository contains a system for detecting and correcting hallucinations in text using Knowledge Graphs (KG) and Large Language Models (LLMs). The system is implemented in two main components: a prototype implementation and a general pipeline.

## Project Structure

```
.
├── prototype/                 # Prototype implementation
│   ├── app_fixed.py          # Streamlit interface
│   ├── main_pipeline.py      # Main pipeline implementation
│   ├── context_KG.py         # Knowledge Graph context retrieval
│   ├── correct_sentence.py   # Sentence correction logic
│   ├── vecatara.py          # Vectara model hallucination detection
│   ├── extract_triplets.py   # Triple extraction from text
│   └── construct_sentence.py # Triplets to Sentence construction 
│
└── general_pipeline/         # General pipeline implementation
    ├── general_interface.py  # General purpose interface
    └── general_pipeline.py   # General pipeline implementation
```

## Features

### Prototype Implementation
- Streamlit-based user interface for interactive testing
- Knowledge Graph-based fact verification
- Triple extraction from text
- Hallucination detection and correction
- Vector search capabilities
- Sentence construction and correction

### General Pipeline
- More generalized implementation of the pipeline
- Enhanced error handling and robustness
- Improved scalability for different use cases
- Flexible interface for various input types

## Requirements

```bash
streamlit>=1.32.0
pandas>=2.2.0
python-dotenv>=1.0.0
neo4j>=5.17.0
transformers>=4.38.0
torch>=2.2.0
google-generativeai>=0.3.2
langchain>=0.1.0
langchain-community>=0.0.10
langchain-core>=0.1.10
openai>=1.12.0
numpy>=1.24.0
scikit-learn>=1.4.0
networkx>=3.2.0
tqdm>=4.66.0
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

## Usage

### Prototype Interface
To run the prototype interface:
```bash
cd prototype
streamlit run app_fixed.py
```

### General Pipeline
To use the general pipeline:
```bash
cd general_pipeline
python general_interface.py
```

## Components

### Prototype Components

1. **Interface (`interface.py`)**
   - Streamlit-based user interface
   - Interactive testing environment
   - Real-time feedback and visualization

2. **Main Pipeline (`main_pipeline.py`)**
   - Core pipeline implementation
   - Integration of all components
   - Error handling and logging

3. **Knowledge Graph Context (`context_KG.py`)**
   - Neo4j database integration
   - Graph query handling
   - Context management

4. **Sentence Correction (`correct_sentence.py`)**
   - Hallucination detection
   - Fact verification
   - Text correction

5. **Vector Search (`vecatara.py`)**
   - Vector-based similarity search
   - Semantic matching
   - Context retrieval

6. **Triple Extraction (`extract_triplets.py`)**
   - Text parsing
   - Triple extraction
   - Entity recognition

7. **Sentence Construction (`construct_sentence.py`)**
   - Natural language generation
   - Sentence structure handling
   - Grammar correction

### General Pipeline Components

1. **General Interface (`general_interface.py`)**
   - Flexible interface for various use cases
   - Enhanced error handling
   - Improved user experience

2. **General Pipeline (`general_pipeline.py`)**
   - Scalable pipeline implementation
   - Modular architecture
   - Extended functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This software is licensed under the GNU General Public License v3.0.

## Contact

gmedhat42@gmail.com
