import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CACHE = {}

logger = logging.getLogger("translator")
logger.setLevel(logging.INFO)

MODEL_MAP = {
    "incoder": "facebook/incoder-1B",
    "codet5": "Salesforce/codet5-base",
    "llama-code": "path_or_hub_id_for_llama_code"
}

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_name="incoder"):
    global MODEL_CACHE

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")

    # Return cached model if already loaded
    if model_name in MODEL_CACHE:
        logger.info(f"Using cached model '{model_name}'")
        return MODEL_CACHE[model_name]

    model_id = MODEL_MAP[model_name]
    logger.info(f"Loading model '{model_name}' for the first time...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    model.to(device)
    model.eval()

    # Cache it
    MODEL_CACHE[model_name] = (tokenizer, model)
    logger.info(f"Model '{model_name}' loaded and cached on {device}")

    return tokenizer, model