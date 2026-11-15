# Atelier — Multi-Modal Retrieval-Augmented Generation (RAG)

Application **Streamlit** de **RAG multimodal** : on pose une question **en texte** sur des
photos de véhicules, l'application retrouve les images les plus pertinentes avec
**OpenCLIP + ChromaDB**, les affiche, puis demande à un **modèle multimodal gratuit**
(**Nemotron 3.5 Super VL** via OpenRouter ou **Gemma 4** via Google AI Studio) de répondre
à la question **en voyant réellement ces images**.

> Sujet de l'atelier : [`RAG Multimodal - OpenCLIP Nemotron Gemma.pdf`](<RAG Multimodal - OpenCLIP Nemotron Gemma.pdf>)

---

## Sommaire

1. [Le problème résolu](#le-problème-résolu)
2. [Fonctionnalités](#fonctionnalités)
3. [Architecture du pipeline](#architecture-du-pipeline)
4. [Structure du projet](#structure-du-projet)
5. [Choix techniques](#choix-techniques)
6. [Prérequis](#prérequis)
7. [Installation](#installation)
8. [Clés API](#clés-api)
9. [Utilisation](#utilisation)
10. [Fonctionnement détaillé du code](#fonctionnement-détaillé-du-code)
11. [Exemple de résultat](#exemple-de-résultat)
12. [Limites et pièges](#limites-et-pièges)
13. [Dépannage](#dépannage)
14. [Pistes d'amélioration](#pistes-damélioration)

---

## Le problème résolu

Un LLM classique ne connaît que son corpus d'entraînement : il ne peut pas répondre à
« quelle est cette voiture rouge ? » sur **vos** images. Deux briques sont donc combinées :

- **Retrieval** — retrouver les bons documents (ici des images) dans une base locale ;
- **Augmented Generation** — donner ces documents au modèle au moment de la question,
  pour qu'il réponde à partir de leur contenu et non de ses souvenirs.

La particularité ici est que la recherche est **multimodale** : la question est du texte,
les documents sont des images, et les deux sont comparés dans **le même espace
vectoriel** grâce à CLIP. C'est ce qui permet de chercher « une voiture sportive rouge »
sans aucune annotation ni étiquette sur les photos.

## Fonctionnalités

- 🔎 **Recherche texte → images** sans mots-clés ni métadonnées : similarité sémantique
  dans un espace joint texte/image (CLIP).
- 🖼️ **Indexation incrémentale** : seules les nouvelles images sont encodées, les
  précédentes ne sont pas recalculées (identifiants stables `img_<nom_du_fichier>`).
- 🎛️ **Nombre de résultats réglable** : 3 par défaut, jusqu'à **toutes** les images
  indexées (curseur dans la barre latérale).
- 🧠 **Deux VLM gratuits au choix** : Nemotron VL (OpenRouter) ou Gemma 4 (Google AI
  Studio), sélectionnables à chaud dans la barre latérale.
- 🔁 **Réindexation en un clic** : bouton « Réindexer les images » dans l'interface.
- 💾 **Persistance locale** : la base ChromaDB survit aux redémarrages, aucun serveur ni
  compte cloud n'est nécessaire pour la partie recherche.
- 🔐 **Aucune clé dans le code** : les secrets sont lus depuis `.env`, ignoré par Git.

## Architecture du pipeline

```
                    ┌──────────────────────────────────────────────┐
                    │  INDEXATION (une fois, puis incrémentale)     │
                    │                                              │
  images/*.jpg ────▶│  OpenCLIP ViT-B-32 (LAION-2B)  ──▶ vecteur    │
                    │  normalisé (512 dims)                        │
                    │        │                                     │
                    │        ▼                                     │
                    │  ChromaDB (collection "vehicules", cosinus)  │
                    └──────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────┐
                    │  REQUÊTE                                      │
                    │                                              │
  question ────────▶│  OpenCLIP encode_text  ──▶ vecteur 512 dims  │
                    │        │                                     │
                    │        ▼                                     │
                    │  ChromaDB.query(top-k) ──▶ métadonnées +     │
                    │                            distances         │
                    │        │                                     │
                    │        ├──▶ affichage des k images           │
                    │        │                                     │
                    │        ▼                                     │
                    │  HumanMessage(texte + images base64)         │
                    │        │                                     │
                    │        ▼                                     │
                    │  Nemotron VL (OpenRouter) / Gemma 4 (Google) │
                    │        │                                     │
                    │        ▼                                     │
                    │  Réponse en langage naturel                  │
                    └──────────────────────────────────────────────┘
```

Deux fichiers seulement portent la logique :

| Fichier | Rôle |
|---|---|
| `src/projet_multimodal/index.py` | indexation des images + recherche des images les plus proches (*retriever*) |
| `src/projet_multimodal/app_streamlit.py` | appel du modèle multimodal + interface web |

## Structure du projet

```
Multi_Rag/                      ← racine du projet (= projet_multimodal)
├── images/                     photos de véhicules : 1.jpg, 2.jpg, 3.jpg …
├── chroma_image_db/            base vectorielle persistante (créée automatiquement, ignorée par Git)
├── .env                        clés API (ignoré par Git)
├── .env.example                modèle de fichier .env
├── .gitignore
├── pyproject.toml              dépendances du projet (uv)
├── requirements.txt            mêmes dépendances pour pip
├── README.md
└── src/projet_multimodal/
    ├── __init__.py
    ├── index.py                indexation + recherche
    └── app_streamlit.py        VLM + interface Streamlit
```

`PROJECT_ROOT` est calculé depuis l'emplacement du fichier
(`Path(__file__).resolve().parents[2]`), donc l'application fonctionne quel que soit le
dossier depuis lequel on la lance, et `images/` comme `chroma_image_db/` sont toujours
résolus à la racine du projet.

## Choix techniques

| Brique | Choix | Pourquoi |
|---|---|---|
| Encodeur image/texte | **OpenCLIP `ViT-B-32`**, poids `laion2b_s34b_b79k` | open source, léger (~600 Mo), entraîné sur LAION-2B : excellent rapport qualité/coût, tourne sur CPU |
| Base vectorielle | **ChromaDB** en mode `PersistentClient` | stockage sur disque sans serveur, recherche ANN (HNSW), métadonnées intégrées |
| Métrique | **cosinus** (`hnsw:space="cosine"`) | adaptée aux embeddings CLIP normalisés : insensible à la norme, compare l'orientation |
| Cache du modèle | `functools.lru_cache` | évite de recharger 600 Mo de poids à chaque appel |
| Cache applicatif | `st.cache_resource` | un seul client LLM et une seule indexation par session Streamlit |
| Orchestration | **LangChain** (`ChatOpenAI`, `ChatGoogleGenerativeAI`, `HumanMessage`) | même interface pour deux fournisseurs, messages multimodaux standardisés |
| Modèles | **Nemotron 3.5 Super VL** (OpenRouter) et **Gemma 4 31B** (Google AI Studio) | gratuits, capables de lire des images |

Pourquoi normaliser les vecteurs ? Chaque embedding est divisé par sa norme
(`features /= features.norm(dim=-1, keepdim=True)`), ce qui rend la similarité cosinus
fiable et la distance ChromaDB interprétable : `distance = 1 − similarité`.

## Prérequis

- **Python ≥ 3.10** (testé avec 3.13) ;
- environ **2 Go d'espace disque** (poids CLIP + dépendances) ;
- une connexion Internet au premier lancement (téléchargement des poids OpenCLIP) ;
- une clé **OpenRouter** et/ou **Google AI Studio** (gratuites) pour la partie génération.

## Installation

### Avec `uv` (recommandé, c'est ce que décrit le sujet)

```bash
uv venv
.venv\Scripts\activate          # Windows PowerShell / CMD
source .venv/bin/activate       # Linux / macOS
uv sync                         # installe les dépendances de pyproject.toml
```

### Sans `uv` (pip)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> La partie recherche tourne **sur CPU** ; aucune carte graphique n'est requise.

## Clés API

Copier le modèle puis renseigner les valeurs :

```bash
copy .env.example .env          # Windows
cp .env.example .env            # Linux / macOS
```

```dotenv
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxxxxxxxx
```

- `OPENROUTER_API_KEY` → https://openrouter.ai/keys (modèle Nemotron)
- `GOOGLE_API_KEY` → https://aistudio.google.com/app/apikey (modèle Gemma 4)

`.env` est listé dans `.gitignore` : les clés ne sont jamais publiées.

## Utilisation

```bash
# 1. Indexer les images (à relancer après ajout de nouvelles photos)
uv run python src/projet_multimodal/index.py
# ou : python src/projet_multimodal/index.py

# 2. Lancer l'interface
uv run streamlit run src/projet_multimodal/app_streamlit.py
# ou : python -m streamlit run src/projet_multimodal/app_streamlit.py
```

Ouvrir ensuite **http://localhost:8501** (`--server.port 8502` si le port est déjà pris).

Dans la barre latérale :

- **Modèle multimodal** — bascule entre Nemotron et Gemma 4 ;
- **Nombre d'images à retrouver** — de 1 à toutes les images indexées (3 par défaut) ;
- **🔄 Réindexer les images** — ajoute uniquement les nouvelles images du dossier.

Puis saisir une question, par exemple :

> « Quelle est la voiture rouge sportive ? Décris-la. »

## Fonctionnement détaillé du code

### `index.py` — indexation + recherche

**Configuration**

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "images"
DB_DIR = PROJECT_ROOT / "chroma_image_db"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
```

**Chargement unique du modèle** (`@lru_cache`) : `create_model_and_transforms` renvoie
`(modèle, prétraitement d'entraînement ignoré, prétraitement d'inférence)`.
`model.eval()` bascule le réseau en mode inférence (pas de dropout, pas d'entraînement).

**Collection ChromaDB** (`@lru_cache`) : `PersistentClient(path=chroma_image_db)` +
`get_or_create_collection("vehicules", metadata={"hnsw:space": "cosine"})`. La collection
est créée au premier lancement, rechargée ensuite.

**Embeddings**

```python
@torch.no_grad()
def embed_image(img):  # image → 512 nombres
    model, preprocess, _ = load_clip_model()
    features = model.encode_image(preprocess(img.convert("RGB")).unsqueeze(0))
    features /= features.norm(dim=-1, keepdim=True)
    return features[0].tolist()
```

- `@torch.no_grad()` : pas de gradient, donc plus rapide et moins gourmand ;
- `img.convert("RGB")` : CLIP exige 3 canaux (une image en niveaux de gris ou avec
  transparence casserait le prétraitement) ;
- `preprocess` : redimensionne en 224×224 et normalise les pixels ;
- `unsqueeze(0)` : ajoute la dimension « lot » (batch de 1 image) ;
- `embed_text` fait exactement la même chose côté texte avec `tokenizer([text])`.

**Indexation incrémentale**

```python
existing_ids = set(collection.get(include=[])["ids"])
for file_path in sorted(IMAGES_DIR.iterdir()):
    doc_id = f"img_{file_path.stem}"          # 40.jpg → img_40
    if file_path.suffix.lower() not in IMAGE_EXTENSIONS or doc_id in existing_ids:
        continue
```

L'identifiant ne dépend que du nom du fichier : ajouter des images ne réindexe pas les
anciennes, et l'opération est idempotente. Chaque ajout stocke `filename` et `path` en
métadonnées, ce qui permet de réafficher l'image depuis ChromaDB.

**Recherche**

```python
collection.query(
    query_embeddings=[embed_text(query)],
    n_results=n_results,
    include=["metadatas", "distances"],
)
```

ChromaDB renvoie **une liste par requête** (même s'il n'y en a qu'une), d'où la lecture
systématique de l'indice `[0]` : `results["metadatas"][0]`, `results["distances"][0]`.
`distance = 1 − similarité cosinus`, donc **plus la distance est petite, plus l'image est
proche** ; l'interface affiche `1 - dist` comme score de similarité.

### `app_streamlit.py` — VLM + interface

**Clients LLM** (`@st.cache_resource`) : un seul client par modèle, réutilisé entre les
interactions.

```python
if provider == "google":
    return ChatGoogleGenerativeAI(model=model_id, google_api_key=..., temperature=0)
return ChatOpenAI(model=model_id, base_url="https://openrouter.ai/api/v1",
                  api_key=..., temperature=0)
```

OpenRouter expose une API **compatible OpenAI**, d'où `ChatOpenAI` + `base_url`.
`temperature=0` rend les réponses plus stables. Ajouter un modèle = ajouter une ligne
dans le dictionnaire `MODELS` (`nom affiché → (fournisseur, identifiant)`).

**Message multimodal** : les images sont lues en binaire, encodées en base64 et injectées
dans une URL `data:` ; le message contient le texte de la question puis les images :

```python
content = [{"type": "text", "text": user_question}]
for path in image_paths:
    content.append({"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}})
return load_llm(provider, model_id).invoke([HumanMessage(content=content)]).text
```

`.text` récupère uniquement la réponse finale (sans le raisonnement interne du modèle).

**Interaction** : `init_index()` (mise en cache) indexe les nouvelles images au premier
chargement, le curseur définit `n_results`, les images sont affichées en grille de 4
colonnes avec leur score, puis la question et les images sont envoyées au modèle choisi.

## Exemple de résultat

Question : « Quelle est la voiture rouge sportive ? Décris-la. »

Images retrouvées : `156.jpg`, `140.jpg`, `99.jpg` — distances cosinus
`0.748 / 0.753 / 0.755` (soit des similarités `0.25 / 0.25 / 0.24`).

Sortie brute de `search_images("a red sports car")` :

```python
{
  "ids": [["img_156", "img_140", "img_99"]],
  "metadatas": [[
      {"filename": "156.jpg", "path": "...\\images\\156.jpg"},
      {"filename": "140.jpg", "path": "...\\images\\140.jpg"},
      {"filename": "99.jpg",  "path": "...\\images\\99.jpg"}
  ]],
  "distances": [[0.748, 0.753, 0.755]]
}
```

Réponse type de Gemma 4 : la voiture la plus « sportive » est une Audi rouge vif
(crossover/SUV compact), avec grandes jantes en alliage, calandre à quatre anneaux et
phares effilés ; la deuxième image montre une Rolls-Royce, orientée luxe et prestige
plutôt que sport.

## Limites et pièges

- **Les modèles gratuits ont un quota.** En cas d'erreur **429**, réessayer plus tard ou
  changer de modèle dans la barre latérale.
- **Chaque image envoyée coûte des tokens.** Demander « toutes les images » est pratique
  pour parcourir la base, mais très coûteux (et souvent refusé) côté VLM : garder un
  `top-k` raisonnable (3 à 6) pour la génération.
- **ChromaDB renvoie des listes imbriquées** : toujours lire `[0]`.
- **Chemins absolus en métadonnées** : si le dossier du projet est déplacé, les chemins
  stockés deviennent invalides → supprimer `chroma_image_db/` et réindexer.
- **Renommer une image** crée un nouvel identifiant : l'ancienne entrée reste dans la
  base tant que la collection n'est pas recréée.
- **CLIP ViT-B-32 est petit** : très bon pour « voiture rouge » ou « SUV noir », plus
  fragile sur des détails fins (modèle de voiture, texte dans l'image).
- **Base locale mono-utilisateur** : `chroma_image_db/` n'est pas fait pour du partage
  concurrent ; il est ignoré par Git et se régénère avec `index.py`.
- **Premier lancement lent** : téléchargement des poids OpenCLIP (~600 Mo) puis encodage
  de toutes les images.

## Dépannage

| Symptôme | Cause / solution |
|---|---|
| `ValueError: OPENROUTER_API_KEY not found` | `.env` absent ou vide → copier `.env.example` et renseigner les clés |
| Erreur **429** / `rate limit` | quota du modèle gratuit → attendre ou changer de modèle |
| `Port 8501 is already in use` | `--server.port 8502` |
| `FileNotFoundError ... images` | le dossier `images/` doit être à la racine du projet |
| Résultats identiques après ajout d'images | cliquer sur « 🔄 Réindexer les images » ou relancer `index.py` |
| `uv sync` bloqué sur le téléchargement de `torch` | la version CUDA (~2,5 Go) est inutile ici : installer `torch` CPU (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) |
| Warning `unauthenticated requests to the HF Hub` | bénin ; définir `HF_TOKEN` pour accélérer le téléchargement |

## Pistes d'amélioration

- Chunking/captions automatiques des images pour un RAG « texte sur images » ;
- ré-encodage par un CLIP plus grand (`ViT-L-14`) si un GPU est disponible ;
- filtres par métadonnées (couleur, marque) en plus de la similarité ;
- historique de conversation (`st.chat_message` + mémoire LangChain) ;
- tests automatisés du retriever (`pytest`) et évaluation du recall@k ;
- déploiement Streamlit Community Cloud avec secrets gérés par la plateforme.

---

Projet réalisé dans le cadre de l'atelier **Multi-Modal RAG** — OpenCLIP, ChromaDB,
Nemotron 3.5 Super VL et Gemma 4.
