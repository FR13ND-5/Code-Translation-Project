from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "facebook/incoder-1B"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if model is None:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.eval()

    print("Model loaded successfully.")
    return tokenizer, model
