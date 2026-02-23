import subprocess
import os
from config import GENERATED_CODE_PATH, OUTPUT_DIR


# remove ```python markdown from LLM output
def clean_code(code: str):
    code = code.replace("```python", "")
    code = code.replace("```", "")
    return code.strip()

def run_code(code):

    # ensure outputs folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    code = clean_code(code)

    with open(GENERATED_CODE_PATH, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python", GENERATED_CODE_PATH],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr

    except Exception as e:
        return False, str(e)


    # clean LLM code
    code = clean_code(code)

    # save generated script
    with open(GENERATED_CODE_PATH, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python", GENERATED_CODE_PATH],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr

    except Exception as e:
        return False, str(e)
