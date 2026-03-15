from .model_loader import load_model
from .prompts import build_prompt
import torch


def translate_code(source_lang, target_lang, code):

    tokenizer , model = load_model()
    code = code.strip()
    prompt = build_prompt(source_lang, target_lang, code)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )

    translated_code = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated_code
