from core.researcher import generate_hypothesis
from core.coder import generate_code
from core.executor import run_code
from config import MAX_RETRIES


def run_ai_scientist():

    dataset_info = "CSV dataset classification problem"
    logs = []

    logs.append("🔬 AI Scientist started")

    # researcher
    hypothesis = generate_hypothesis(dataset_info)
    logs.append("🧠 Hypothesis:\n" + hypothesis)

    # coder
    code = generate_code(hypothesis)
    logs.append("💻 Code generated")

    # executor loop
    for attempt in range(MAX_RETRIES):
        logs.append(f"\n⚙️ Running attempt {attempt + 1}")
        success, output = run_code(code)

        if success:
            logs.append("✅ Success:\n" + output)
            break
        else:
            logs.append("❌ Error:\n" + output)
            logs.append("🔁 Sending error to coder to fix...")

            fix_prompt = f"""
Fix this Python ML code.

Error:
{output}

Code:
{code}

Return only fixed code.
"""
            code = generate_code(fix_prompt)

    return "\n".join(logs)


if __name__ == "__main__":
    print(run_ai_scientist())
