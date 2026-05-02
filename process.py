import pandas as pd
import time
import os
import sys
from groq import Groq

# ───────── CONFIG ─────────
MODEL = "llama-3.3-70b-versatile"
SLEEP = 10
MAX_RETRIES = 3

# 10 datasets + 10 API keys
DATASETS = [
    ("train_part_1.csv", "output_part_1.csv", "GROQ_API_KEY_1"),
    ("train_part_2.csv", "output_part_2.csv", "GROQ_API_KEY_2"),
    ("train_part_3.csv", "output_part_3.csv", "GROQ_API_KEY_3"),
    ("train_part_4.csv", "output_part_4.csv", "GROQ_API_KEY_4"),
    ("train_part_5.csv", "output_part_5.csv", "GROQ_API_KEY_5"),
    ("train_part_6.csv", "output_part_6.csv", "GROQ_API_KEY_6"),
    ("train_part_7.csv", "output_part_7.csv", "GROQ_API_KEY_7"),
    ("train_part_8.csv", "output_part_8.csv", "GROQ_API_KEY_8"),
    ("train_part_9.csv", "output_part_9.csv", "GROQ_API_KEY_9"),
    ("train_part_10.csv", "output_part_10.csv", "GROQ_API_KEY_10"),
]
# ─────────────────────────

# ───────── LOGGING ─────────
def log(msg, file="logs.txt"):
    print(msg)
    sys.stdout.flush()
    with open(file, "a") as f:
        f.write(msg + "\n")

# ───────── PROMPT ─────────
def build_prompt(prompt, answer):
    return f"""
You are generating a high-quality reasoning dataset.

Question:
{prompt}

Correct Answer:
{answer}

Return EXACTLY in this format:

CoT:
Domain: <identify domain>

Step 1: ...
Step 2: ...
Step 3: ...
... (use as many steps as needed)

Final Answer:
{answer}

IMPORTANT:
- CoT must contain ONLY reasoning
- Do NOT include final answer inside CoT
- Steps must be short and logical
- No extra text outside this format
"""

# ───────── API CALL ─────────
def generate(client, prompt, answer):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Follow format strictly."},
                    {"role": "user", "content": build_prompt(prompt, answer)}
                ],
                temperature=0.2,
                max_completion_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            log(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(10 * (attempt + 1))

    return None

# ───────── PROCESS ONE FILE ─────────
def process_dataset(input_file, output_file, key_env):

    API_KEY = os.getenv(key_env)

    if not API_KEY:
        log(f"❌ Missing API key: {key_env}")
        return

    client = Groq(api_key=API_KEY)

    log(f"\n🚀 Processing {input_file}")

    # Load and clean structure
    df = pd.read_csv(input_file, engine="python")
    df = df[["id", "prompt", "answer"]]

    df["cot"] = ""
    df["final_answer"] = ""
    df["final_answer"] = df["final_answer"].astype(str)

    total = len(df)
    log(f"📂 Loaded {total} rows from {input_file}")

    for i, row in df.iterrows():

        if str(df.loc[i, "cot"]).strip() not in ["", "nan"]:
            continue

        log(f"[{input_file}] Row {i}/{total}")

        output = generate(client, row["prompt"], row["answer"])

        if output is None:
            log("❌ Failed, skipping")
            continue

        # -------- CLEAN OUTPUT --------
        if "Final Answer:" in output:
            cot_part = output.split("Final Answer:")[0]
        else:
            cot_part = output

        cot_part = cot_part.strip()

        # Save reasoning
        df.loc[i, "cot"] = cot_part

        # Save answer (FIXED TYPE)
        df.loc[i, "final_answer"] = str(row["answer"])

        # Save only required columns
        df_clean = df[["id", "prompt", "cot", "final_answer"]]
        df_clean.to_csv(output_file, index=False)

        log(f"💾 Saved {output_file} row {i}")

        time.sleep(SLEEP)

    log(f"✅ Completed {output_file}")

# ───────── RUN ALL DATASETS ─────────
for inp, out, key in DATASETS:
    process_dataset(inp, out, key)

log("\n🎯 ALL DATASETS COMPLETED")
