from codebleu import calc_code_bleu
import subprocess
import logging
import tempfile
import os

# Create logger
logger = logging.getLogger("translator")
logger.setLevel(logging.INFO)


def evaluate_translation(reference_code, generated_code, lang="python"):
    try:
        metrics = calc_code_bleu(
            ref_file_path=None,
            hyp_file_path=None,
            ref_strs=[reference_code],
            hyp_strs=[generated_code],
            lang=lang
        )
        return metrics
    except Exception as e:
        # Log failure but return empty metrics
        import logging
        logging.getLogger("translator").warning(f"CodeBLEU evaluation failed: {str(e)}")
        return {}

def compile_code_cpp(code):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as f:
        f.write(code.encode())
        temp_file = f.name

    result = subprocess.run(
        ["g++", temp_file, "-o", f"{temp_file}.out"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    # Log compilation errors if any
    if result.returncode != 0:
        logger.warning(
            f"C++ compilation failed: {result.stderr.decode()}"
        )

    # Clean up temporary files
    os.remove(temp_file)
    if os.path.exists(f"{temp_file}.out"):
        os.remove(f"{temp_file}.out")
        
    return result.returncode == 0