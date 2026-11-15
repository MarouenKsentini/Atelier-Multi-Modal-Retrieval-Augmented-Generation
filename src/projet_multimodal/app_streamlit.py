"""STREAMLIT APP — question → images les plus proches (index.py) →
réponse d'un modèle multimodal gratuit :
  - Nemotron VL via OpenRouter (OPENROUTER_API_KEY)
  - Gemma 4 via l'API Google AI Studio (GOOGLE_API_KEY)

Lancement :  uv run streamlit run src/projet_multimodal/app_streamlit.py
"""

import base64
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent))  # pour importer index.py voisin
from index import (  # noqa: E402
    DEFAULT_N_RESULTS,
    PROJECT_ROOT,
    get_chroma_collection,
    index_images,
    search_images,
)

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(layout="wide", page_title="Recherche Images + VLM")
load_dotenv(PROJECT_ROOT / ".env", override=True)  # OPENROUTER_API_KEY, GOOGLE_API_KEY

# Modèles multimodaux gratuits : nom affiché → (fournisseur, identifiant du modèle)
MODELS = {
    "Nemotron 3.5 Super VL (NVIDIA)": ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    "Gemma 4 31B (Google)": ("google", "gemma-4-31b-it"),
}


# @st.cache_resource : un client par modèle, créé une seule fois puis réutilisé
@st.cache_resource
def load_llm(provider: str, model_id: str):
    if provider == "google":  # API native Google AI Studio
        return ChatGoogleGenerativeAI(
            model=model_id, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0
        )
    return ChatOpenAI(  # OpenRouter : API compatible OpenAI
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


@st.cache_resource(show_spinner="Chargement de CLIP et indexation des images…")
def init_index():
    return index_images()  # n'ajoute que les nouvelles images


# =========================================================
# APPEL DU MODÈLE MULTIMODAL
# =========================================================
# Type MIME réel de l'image : le dossier mélange .jpg, .jpeg, .png et .webp
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:  # rb : read binary
        return base64.b64encode(f.read()).decode("utf-8")


def ask_llm(user_question: str, image_paths: list[str], provider: str, model_id: str) -> str:
    content = [{"type": "text", "text": user_question}]
    for path in image_paths:
        mime = MIME_TYPES.get(Path(path).suffix.lower(), "image/jpeg")
        content.append(  # image en base64 dans une URL "data:"
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encode_image(path)}"},
            }
        )
    return load_llm(provider, model_id).invoke([HumanMessage(content=content)]).text


# =========================================================
# INTERFACE
# =========================================================
def main():
    init_index()

    total_images = get_chroma_collection().count()

    with st.sidebar:
        st.header("Modèle multimodal")
        model_name = st.radio("Choisis le modèle :", list(MODELS))
        provider, model_id = MODELS[model_name]

        st.divider()
        st.header("Recherche")
        # search_images(..., n_results=DEFAULT_N_RESULTS) par défaut → ici on peut aller jusqu'à toutes
        max_images = max(1, total_images)
        n_images = st.slider(
            "Nombre d'images à retrouver",
            min_value=1,
            max_value=max_images,
            value=min(DEFAULT_N_RESULTS, max_images),
            help="Monte le curseur jusqu'au maximum pour retrouver toutes les images indexées.",
        )
        st.caption(f"{total_images} images indexées")
        st.caption("⚠️ Plus tu envoies d'images au modèle, plus la réponse est lente (et limitée en tokens).")

        if st.button("🔄 Réindexer les images", type="primary"):
            with st.spinner("Indexation en cours..."):
                st.success(f"{index_images()} nouvelles images indexées !")

    st.title(f"🔍 Recherche d'Images + {model_name.split(' (')[0]}")

    user_question = st.text_input(
        "Pose ta question :",
        placeholder="Décris les voitures sportives rouges ou quelles sont les SUV noires ?",
    )
    if not user_question:
        return

    with st.spinner("Recherche des images les plus pertinentes..."):
        results = search_images(user_question, n_results=n_images)
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    st.subheader(f"🖼️ Images retrouvées ({len(metadatas)})")
    # affichage en grille de 4 colonnes (lisible même avec toutes les images)
    COLONNES = 4
    for start in range(0, len(metadatas), COLONNES):
        cols = st.columns(COLONNES)
        for col, meta, dist in zip(
            cols,
            metadatas[start : start + COLONNES],
            distances[start : start + COLONNES],
        ):
            col.image(
                meta["path"],
                caption=f"{meta['filename']} — similarité {1 - dist:.2f}",
                width="stretch",
            )

    with st.spinner(f"Analyse avec {model_name}..."):
        response = ask_llm(user_question, [m["path"] for m in metadatas], provider, model_id)

    st.subheader(f"Réponse de {model_name}")
    st.markdown(response)


main()
