from fastapi import FastAPI
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are a denial-reason classifier. "
    "Return one of: eligibility, coding, timely filing, medical necessity, prior auth."
)

app = FastAPI(title="DenialRx Tiny Pilot")

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/classify")
def classify(text: str):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",            # fallback: "gpt-3.5-turbo-0125"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0,
    )
    return {"denial_reason": response.choices[0].message.content.strip().lower()}
