# File: vectfinal.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
import torch
import time
from typing import List, Dict, Union, Optional

hf_logging.set_verbosity_error()

MODEL_NAME = "vectara/hallucination_evaluation_model"
TOKENIZER_NAME = "google/flan-t5-base" # Keep tokenizer name, might be needed internally or for other purposes, though predict might handle it.
HALLUCINATION_THRESHOLD = 0.5 # You might need to adjust this based on testing with direct scores

# --- Load Model and Tokenizer Directly ---
hf_model = None
hf_tokenizer = None
try:
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    # Check for MPS (Apple Silicon) - Optional but good practice
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Loading model: {MODEL_NAME}")
    # Load the model and move it to the determined device
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).to(device)
    hf_model.eval() # Set model to evaluation mode
    print("Hugging Face model and tokenizer initialized successfully.")

except Exception as e:
    print(f"ERROR: Failed to initialize Hugging Face model/tokenizer: {e}")
    hf_model = None # Ensure it's None on failure
    hf_tokenizer = None

# --- Modified Detection Function ---
def detect_hallucinations_batch(
    model, # Expect the loaded model object directly
    premises: List[str],
    hypotheses: List[str]
    ) -> List[Dict[str, Union[str, float, bool, None]]]:

    if model is None: # Check if the model object exists
        print("ERROR: Hallucination detection model not initialized.")
        # Return results indicating error for each pair
        return [{"premise": p, "hypothesis": h, "score": None, "is_hallucinated": None, "error": "Model not loaded"}
                for p, h in zip(premises, hypotheses)]

    if not premises or not hypotheses or len(premises) != len(hypotheses):
        print("ERROR: Input premise/hypothesis lists are empty or have mismatched lengths.")
        # Return results indicating error for each potential pair structure
        max_len = max(len(premises) if premises else 0, len(hypotheses) if hypotheses else 0)
        premises = premises if premises else [""] * max_len
        hypotheses = hypotheses if hypotheses else [""] * max_len
        return [{"premise": p, "hypothesis": h, "score": None, "is_hallucinated": None, "error": "Input list mismatch"}
                for p, h in zip(premises, hypotheses)]

    # Prepare pairs as list of tuples for model.predict
    pairs = list(zip(premises, hypotheses))

    results_list = []
    try:
        print(f"Running prediction on {len(pairs)} pairs...")
        start_time = time.time()
        # Use torch.no_grad() for inference efficiency
        with torch.no_grad():
             # Call the model's specific predict method
            scores_tensor = model.predict(pairs)
        end_time = time.time()
        print(f"Prediction took {end_time - start_time:.2f} seconds.")

        # Move tensor to CPU (if not already), detach, convert to numpy, then list
        scores = scores_tensor.cpu().numpy().tolist()

        if len(scores) != len(pairs):
            print(f"ERROR: Mismatch between input pairs ({len(pairs)}) and output scores ({len(scores)})")
            # Handle error - return error status for all
            return [{"premise": p, "hypothesis": h, "score": None, "is_hallucinated": None, "error": "Prediction output mismatch"}
                    for p, h in pairs]

        print("Processing prediction results...")
        for i, (prem, hypo) in enumerate(pairs):
            consistency_score = scores[i] # The direct score from the model

            # Check if score is valid number, handle potential NaNs or Infs if necessary
            if not isinstance(consistency_score, (int, float)):
                 print(f"Warning: Invalid score type received for pair {i}: {consistency_score}")
                 is_hallucinated = None # Mark as error/unknown
                 consistency_score = None # Set score to None
            else:
                 # Apply threshold: Higher score means MORE consistent (based on your test.py)
                 is_hallucinated = consistency_score < HALLUCINATION_THRESHOLD

            results_list.append({
                "premise": prem,
                "hypothesis": hypo,
                "score": consistency_score,  # Changed from consistency_score to score
                "is_hallucinated": is_hallucinated # True if score is LOW
            })

    except Exception as e:
        print(f"  - Error during model.predict batch prediction: {e}")
        # Return error status for all items in the batch
        return [{"premise": p, "hypothesis": h, "score": None, "is_hallucinated": None, "error": str(e)}
                for p, h in pairs]

    print("Hallucination detection batch completed.")
    return results_list