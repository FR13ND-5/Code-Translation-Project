import re
import logging

logger = logging.getLogger("translator")

# Regex patterns to detect function/method definitions per language
LANGUAGE_PATTERNS = {
    "python": r"((?:@\w+\s*)*def\s+\w+\s*\(.*?\)\s*:\s*)",
    "java": r"((public|private|protected|static|final|synchronized|abstract|\s)+\s+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{)",
    "cpp": r"((unsigned|long|short|static|inline|virtual|constexpr|[\w:&*<>\[\]]+)+\s+\w+\s*\([^)]*\)\s*(const)?\s*\{)"
}

def split_into_functions(code, language="python"):
    """
    Splits code into logical function/method chunks based on language-specific regex.
    
    Args:
        code (str): The source code to split.
        language (str): Programming language ("python", "java", "cpp").
        
    Returns:
        List[str]: A list of code chunks (functions/methods).
    """
    pattern = LANGUAGE_PATTERNS.get(language.lower())
    if not pattern:
        return [code]

    parts = re.split(pattern, code , flags=re.DOTALL)
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(pattern, part):
            if current:
                chunks.append(current)
            current = part
        else:
            current += "\n" + part

    if current:
        chunks.append(current.strip())

    # Fallback if no functions detected
    if not chunks:
        chunks = [code.strip()]

     # ---- Add logging here ----
    logger.info(f"{len(chunks)} chunks detected for {language} code")

    return chunks