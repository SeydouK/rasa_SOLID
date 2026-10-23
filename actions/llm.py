from typing import List, Optional
import os 
import requests

MODELE = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
TIMEOUT = 15

CONSIGNE = (
    "Tu es l'assistant du Port Autonome de Dakar. Réponds en français, en 2 à 4 phrases, "
    "uniquement à partir des extraits fournis. Si les extraits ne contiennent pas la "
    "réponse, dis-le simplement sans inventer. Ne recopie pas les extraits mot pour mot."
)

if not os.getenv("GEMINI_API_KEY"):
    print("[LLM] GEMINI_API_KEY absente de l'environnement")

def reformuler(question: str, passages: List[str]) -> Optional[str]:
    cle = os.getenv("GEMINI_API_KEY")
    if not cle or not passages:
        return None

    extraits = "\n\n".join(f"[Extrait {i + 1}] {p}" for i, p in enumerate(passages))
    corps = {
        "system_instruction": {"parts": [{"text": CONSIGNE}]},
        "contents": [{
            "role": "user",
            "parts": [{"text": f"Question : {question}\n\nExtraits :\n{extraits}"}],
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
    }
    try:
        r = requests.post(
            URL.format(MODELE),
            headers={"x-goog-api-key": cle, "Content-Type": "application/json"},
            json=corps,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        detail = getattr(getattr(e, "response", None), "text", "")
        print(f"[LLM] erreur : {e} {detail[:300]}")
        return None
