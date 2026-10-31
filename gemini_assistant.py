
import os
import tomllib
from pathlib import Path

import streamlit as st
from google import genai


# --------------------------------------------------
# 1. LOAD GEMINI API KEY
# --------------------------------------------------



def get_gemini_client():

    import os
    import tomllib
    from pathlib import Path
    from google import genai

    api_key = None

    # Streamlit Cloud secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Local secrets.toml
    if not api_key:
        secrets_path = (
            Path(__file__).resolve().parent
            / ".streamlit"
            / "secrets.toml"
        )

        if secrets_path.exists():
            with open(secrets_path, "rb") as file:
                secrets_data = tomllib.load(file)

            api_key = secrets_data.get("GEMINI_API_KEY")

    # Environment variable fallback
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)



# --------------------------------------------------
# 2. GENERATE AI DIAGNOSIS
# --------------------------------------------------

def generate_diagnosis(
    fault_type,
    probability,
    machine_data,
    language="English"
):

    client = get_gemini_client()

    if client is None:
        return (
            "Gemini API key not configured. "
            "The ML prediction is available, but "
            "AI explanation is disabled."
        )

    prompt = f"""
You are PLC Doctor, an industrial maintenance
diagnostic assistant.

Explain the following machine learning prediction
to an industrial technician.

PREDICTED FAULT:
{fault_type}

MODEL CONFIDENCE:
{probability:.2f}%

MACHINE MEASUREMENTS:
{machine_data}

Write your response in {language}.

Use these sections:

1. DIAGNOSIS
2. POSSIBLE CAUSES
3. VERIFICATION STEPS
4. RECOMMENDED ACTIONS
5. SAFETY PRECAUTIONS

Safety rules:
- Never recommend bypassing safety interlocks.
- Never recommend disabling machine protections.
- Never instruct anyone to work on energized equipment.
- Do not present the prediction as a confirmed fault.
- Physical inspection by a qualified technician is required.
- Never invent machine specifications or fault codes.

Keep the response clear and practical.
"""

    try:

        response = client.models.generate_content(
           model="gemini-3.5-flash-lite",
            contents=prompt
        )

        if response.text:
            return response.text

        return "Gemini returned an empty response."

    except Exception as error:

        return f"Gemini API request failed: {error}"
