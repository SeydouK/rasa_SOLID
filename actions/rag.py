import hashlib
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".rag_cache"

MODELE = os.getenv("RAG_MODEL", "intfloat/multilingual-e5-small")

SEUIL = float(os.getenv("RAG_SEUIL", "0.80"))


class MoteurDocs:
    def __init__(self):
        self.morceaux = []   
        self._decouper()

        self.modele = SentenceTransformer(MODELE)
        self.matrice = self._charger_ou_calculer_embeddings()

    def _decouper(self):
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

            for i in range(0, len(phrases_buffer), 3):
                fenetre = phrases_buffer[i:i + 5]
                texte_chunk = " ".join(p for _, p in fenetre)
                if len(texte_chunk) > 60:
                    self.morceaux.append((f"{f.name}, p.{fenetre[0][0]}", texte_chunk))

    def _empreinte(self) -> str:
        h = hashlib.sha256(MODELE.encode())
        for _, t in self.morceaux:
            h.update(t.encode())
        return h.hexdigest()[:16]

    def _charger_ou_calculer_embeddings(self) -> np.ndarray:
        if not self.morceaux:
            return np.empty((0, 0), dtype=np.float32)

        CACHE_DIR.mkdir(exist_ok=True)
        fichier = CACHE_DIR / f"emb_{self._empreinte()}.npy"
        if fichier.exists():
            return np.load(fichier)

        emb = self.modele.encode(
            ["passage: " + t for _, t in self.morceaux],
            normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        ).astype(np.float32)
        np.save(fichier, emb)
        return emb

    def chercher(self, question: str, k: int = 2):
        if not self.morceaux:
            return []
        
        q = self.modele.encode(
            ["query: " + question], normalize_embeddings=True
        ).astype(np.float32)[0]
        scores = self.matrice @ q          
        ordre = scores.argsort()[::-1][:k]
        return [(self.morceaux[i], float(scores[i])) for i in ordre if scores[i] >= SEUIL]


MOTEUR = MoteurDocs()