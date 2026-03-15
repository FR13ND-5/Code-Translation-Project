def build_prompt(source_lang, target_lang, code):
    code = code.strip()  # remove extra whitespace
    prompt = f"Convert the following {source_lang} code to {target_lang}:\n{code}\n{target_lang} code:\n"
    return prompt