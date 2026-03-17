import subprocess
import tempfile
import threading
import functools
import logging
import os

_clang_format_lock = threading.Lock()
_clang_format_process = None

# Create logger
logger = logging.getLogger("translator")
logger.setLevel(logging.INFO)

def _start_clang_format():
    """
    Starts clang-format subprocess if not already running.
    Re-starts if previous process crashed.
    """
    global _clang_format_process
    if _clang_format_process is None or _clang_format_process.poll() is not None:
        try: 
            _clang_format_process = subprocess.Popen(
                ["clang-format", "-style=google"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            logger.warning(f"Failed to start clang-format: {str(e)}")
            _clang_format_process = None

def format_code(code, language):
    """
    Dispatch formatting based on language.
    """
    language = language.lower()
    if language == "python":
        return format_python(code)
    elif language in ["javascript", "typescript"]:
        return format_js(code)
    elif language == "java":
        return format_java(code)
    elif language in ["c","cpp","c++"]:
        return format_cpp(code)
    else:
        return code

def format_python(code):
    import black
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        logger.warning(f"Python formatting failed: {str(e)}")
        return code

def format_js(code):
    try:
        result = subprocess.run(
            ["npx", "prettier", "--parser", "babel"],
            input=code,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            logger.warning(f"JS formatting failed: {result.stderr}")
            return code
        return result.stdout
    except Exception as e:
        logger.warning(f"JS formatting exception: {str(e)}")
        return code

def format_java(code):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".java") as f:
            f.write(code.encode())
            file_path = f.name

        subprocess.run(["google-java-format", "-i", file_path])

        with open(file_path) as f:
            return f.read()

    except Exception as e:
        logger.warning(f"Java formatting failed: {str(e)}")
        return code
    finally:
        try:
            os.remove(file_path)
        except:
            pass
    
@functools.lru_cache(maxsize=512)
def format_cpp(code):
    """
    Formats C/C++ code using a persistent clang-format process.
    Thread-safe with lock. Returns unformatted code on failure.
    """
    global _clang_format_process
    _start_clang_format()

    if _clang_format_process is None:
        logger.warning("C++ formatting skipped: clang-format not available")
        return code

    try:
        with _clang_format_lock:
            # send code + sentinel to clang-format
            _clang_format_process.stdin.write(code + "\n")
            _clang_format_process.stdin.write("// END_OF_CODE\n")
            _clang_format_process.stdin.flush()

            # read untill sentinel
            stdout = ""
            while True:
                line = _clang_format_process.stdout.readline()
                if "// END_OF_CODE" in line:
                    break
                stdout += line

            return stdout
    except Exception as e:
        # Fallback: return unformatted code
        logger.warning("C++ formatting failed", exc_info=True)
        return code 