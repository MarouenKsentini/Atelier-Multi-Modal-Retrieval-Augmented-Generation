"""INDEXATION + RECHERCHE — OpenCLIP + ChromaDB.

Indexer les images :  uv run python src/projet_multimodal/index.py
"""

from functools import lru_cache
from pathlib import Path

import chromadb
import open_clip
import torch
from PIL import Image

# =========================================================
# CONFIGURATION
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/projet_multimodal
IMAGES_DIR = PROJECT_ROOT / "images"
DB_DIR = PROJECT_ROOT / "chroma_image_db"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
# Modèle OpenCLIP : architecture ViT-B-32, poids entraînés sur LAION-2B (open source)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


@lru_cache  # chargé une seule fois
def load_clip_model():
    """Retourne (modèle, prétraitement des images, tokenizer du texte)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model.eval()  # mode inférence
    return model, preprocess, open_clip.get_tokenizer(CLIP_MODEL)


@lru_cache
def get_chroma_collection() -> chromadb.Collection:
    """Collection persistante, similarité cosinus (idéale pour CLIP)."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name="vehicules", metadata={"hnsw:space": "cosine"})


# =========================================================
# EMBEDDINGS (CLIP)
# =========================================================
@torch.no_grad()  # pas de calcul de gradients : on n'entraîne pas le modèle
def embed_image(img: Image.Image) -> list[float]:
    model, preprocess, _ = load_clip_model()
    # CLIP a besoin d'un format standard : RGB (rouge, vert, bleu), redimensionné
    features = model.encode_image(preprocess(img.convert("RGB")).unsqueeze(0))
    features /= features.norm(dim=-1, keepdim=True)  # normalisation (cosinus)
    return features[0].tolist()


@torch.no_grad()
def embed_text(text: str) -> list[float]:
    model, _, tokenizer = load_clip_model()
    features = model.encode_text(tokenizer([text]))
    features /= features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


# =========================================================
# INDEXATION
# =========================================================
def index_images() -> int:
    """Image → embedding CLIP → ChromaDB. Retourne le nombre d'images ajoutées."""
    collection = get_chroma_collection()
    existing_ids = set(collection.get(include=[])["ids"])
    indexed = 0

    for file_path in sorted(IMAGES_DIR.iterdir()):
        doc_id = f"img_{file_path.stem}"  # id stable = nom du fichier
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS or doc_id in existing_ids:
            continue  # pas une image, ou déjà indexée

        with Image.open(file_path) as img:
            embedding = embed_image(img)
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"filename": file_path.name, "path": str(file_path)}],
        )
        indexed += 1

    return indexed


# =========================================================
# RECHERCHE (retriever)
# =========================================================
DEFAULT_N_RESULTS = 9  # nombre d'images renvoyées par défaut (réglable dans l'interface)


def search_images(query: str, n_results: int = DEFAULT_N_RESULTS) -> dict:
    """Question → embedding CLIP → images les plus proches.

    ChromaDB renvoie une liste de résultats par requête, d'où les [0] à la lecture.
    """
    return get_chroma_collection().query(
        query_embeddings=[embed_text(query)],
        n_results=n_results,
        include=["metadatas", "distances"],
    )


def main() -> None:
    print(f"{index_images()} nouvelle(s) image(s) indexée(s)")


if __name__ == "__main__":
    main()
