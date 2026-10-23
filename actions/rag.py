import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SEUIL = 0.15
STOP = ["le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "au", "aux",
        "que", "qui", "quoi", "est", "sont", "pour", "dans", "sur", "par", "ce", "ces",
        "quel", "quelle", "quels", "quelles", "dit", "vos", "votre", "il", "y", "a"]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


class MoteurDocs:
    def __init__(self):
        self.morceaux = []   # (source, texte)

        for f in sorted(DOCS_DIR.glob("*")):
            if f.suffix.lower() not in (".txt", ".md"):
                continue

            contenu = f.read_text(encoding="utf-8")
            page = "?"
            phrases_buffer = []

            for bloc in re.split(r"\n\s*\n", contenu):
                m = re.match(r"\[Page (\d+)\]", bloc)
                if m:
                    page = m.group(1)
                texte = re.sub(r"\[Page \d+\]\n?", "", bloc)
                texte = re.sub(r"\s*\n\s*", " ", texte).strip()
                for ph in re.split(r"(?<=[.!?])\s+", texte):
                    if ph:
                        phrases_buffer.append((page, ph))

            # fenêtres de 5 phrases, avec un pas de 3 (chevauchement de 2)
            for i in range(0, len(phrases_buffer), 3):
                fenetre = phrases_buffer[i:i + 5]
                texte_chunk = " ".join(p for _, p in fenetre)
                if len(texte_chunk) > 60:
                    self.morceaux.append((f"{f.name}, p.{fenetre[0][0]}", texte_chunk))

        self.vec = TfidfVectorizer(
            preprocessor=_norm,
            stop_words=[_norm(s) for s in STOP],
            ngram_range=(1, 2),
        )
        if self.morceaux:
            self.matrice = self.vec.fit_transform([t for _, t in self.morceaux])

    def chercher(self, question: str, k: int = 2):
        if not self.morceaux:
            return []
        scores = cosine_similarity(self.vec.transform([question]), self.matrice)[0]
        ordre = scores.argsort()[::-1][:k]
        return [(self.morceaux[i], scores[i]) for i in ordre if scores[i] >= SEUIL]


MOTEUR = MoteurDocs()