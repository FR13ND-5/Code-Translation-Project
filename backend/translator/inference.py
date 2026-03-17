from .model_loader import get_model, device
from .prompts import build_prompt
from .formatter import format_code
from .chunker import split_into_functions
from .evaluation import evaluate_translation
from .evaluation import compile_code_cpp
import torch
import time
import logging 
import json

# Create logger
logger = logging.getLogger("translator")
logger.setLevel(logging.INFO)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage()
        })

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(JsonFormatter())
logger.addHandler(ch)

def _translate_chunk(tokenizer, model, source_lang, target_lang, code):
    """
    Low-level translation for a single code chunk using passed tokenizer & model
    """
    code = code.strip()
    prompt = build_prompt(source_lang, target_lang, code)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,          # supports long functions
            do_sample=False,             # deterministic output
            temperature=0.2,             # reduces hallucination
            repetition_penalty=1.1,      # avoid repetition
            eos_token_id=tokenizer.eos_token_id
        )

    translated_code = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove the prompt from output
    translated_code = translated_code.replace(prompt, "").strip()

    # Cut off any extra sections if model outputs ###
    if "###" in translated_code:
        translated_code = translated_code.split("###")[0]

    return translated_code


def translate_code(source_lang, target_lang, code, model_name = "incoder"):
    """
    High-level translation: splits large files into chunks,
    translates each chunk, formats code, and returns the final file.
    """
    tokenizer, model = get_model(model_name)
    
    MAX_FILE_SIZE = 1_000_000   #~ 1 MB
    if len(code) > MAX_FILE_SIZE:
        raise ValueError("File too large. Maximum allowed size is 1 MB.")
    
    chunks = split_into_functions(code, language=source_lang.lower())

    BATCH_MAX_TOKENS = 400   # adjust based on your model
    translated_chunks = []
    batch = []
    batch_tokens = 0
    latency_history = []

    for chunk in chunks:
        chunk_tokens = len(tokenizer(chunk)["input_ids"])
        if batch_tokens + chunk_tokens < BATCH_MAX_TOKENS:
            batch.append(chunk)
            batch_tokens += chunk_tokens
        else:
            # Translate current batch
            batch_text = "\n".join(batch)
            try:
                start = time.time()
                translated = _translate_chunk(tokenizer , model , source_lang, target_lang, batch_text)
                end = time.time()
                batch_latency = end - start
                logger.info(f"Batch size tokens={batch_tokens}, latency={batch_latency:.2f}s")

                # -----------------------------
                # Adjust BATCH_MAX_TOKENS dynamically
                # -----------------------------
                latency_history.append(batch_latency)
                if len(latency_history) > 3:
                    latency_history.pop(0)

                avg_latency = sum(latency_history)/len(latency_history)

                # Adjust batch tokens based on average
                if avg_latency > 5:
                    BATCH_MAX_TOKENS = max(100, BATCH_MAX_TOKENS - 50)
                    logger.info(f"Reducing BATCH_MAX_TOKENS to {BATCH_MAX_TOKENS}")
                elif avg_latency < 1:
                    BATCH_MAX_TOKENS += 50
                    logger.info(f"Increasing BATCH_MAX_TOKENS to {BATCH_MAX_TOKENS}")

            except Exception as e:
                logger.warning(f"Batch translation failed",  exc_info=True)
                translated = f"// Batch translation failed: {str(e)}\n{batch_text}"

            try:
                translated = format_code(translated, target_lang)

                # Only for C/C++ translations
                if target_lang.lower() in ["c", "cpp", "c++"]:
                    success = compile_code_cpp(translated)
                    logger.info(f"C++ compile success: {success}")
            except Exception as e:
                logger.warning(f"Batch formatting failed", exc_info=True)
                translated = f"// Batch formatting failed: {str(e)}\n{translated}"

            translated_chunks.append(translated)

            # Start new batch
            batch = [chunk]
            batch_tokens = chunk_tokens

    # Translate remaining batch
    if batch:
        batch_text = "\n".join(batch)
        try:
            translated = _translate_chunk(tokenizer, model ,source_lang, target_lang, batch_text)
        except Exception as e:
            logger.warning(f"Batch translation failed: {str(e)}")
            translated = f"// Batch translation failed: {str(e)}\n{batch_text}"

        try:
            translated = format_code(translated, target_lang)
        except Exception as e:
            logger.warning(f"Batch formatting failed: {str(e)}")
            translated = f"// Batch formatting failed: {str(e)}\n{translated}"

        translated_chunks.append(translated)

    final_translation = "\n\n".join(translated_chunks)

    # -----------------------------
    # CODEBLEU / RESEARCH METRICS
    # -----------------------------
    try:
        metrics = evaluate_translation(code, final_translation, lang=source_lang.lower())
        logger.info(f"CodeBLEU metrics: {metrics}")
    except Exception as e:
        logger.warning(f"CodeBLEU evaluation failed: {str(e)}")

    # C++ compile success check for research
    if target_lang.lower() in ["c", "cpp", "c++"]:
        try:
            success = compile_code_cpp(final_translation)
            logger.info(f"C++ compile success: {success}")
        except Exception as e:
            logger.warning(f"C++ compile check failed: {str(e)}")

    return final_translation

