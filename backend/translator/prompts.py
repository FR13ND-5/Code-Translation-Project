def build_prompt(source_lang, target_lang, code):

    prompt = f"""
### {source_lang}
{code}

### {target_lang}
"""

    return prompt.strip()