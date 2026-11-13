"""Affiche le contenu de l'index FAISS, sans appeler Ollama : python -m script.read_index"""

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS

from config import FAISS_INDEX_PATH


def read_index(index_path=FAISS_INDEX_PATH, limit=5):
    """Charge l'index sans appeler le fournisseur d'embeddings et affiche son contenu."""
    index_path = index_path.resolve()
    faiss_file = index_path / "index.faiss"
    metadata_file = index_path / "index.pkl"
    if not faiss_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"Index FAISS incomplet dans {index_path} (index.faiss et index.pkl requis)."
        )

    # index.pkl est produit par notre propre pipeline. Son chargement utilise pickle.
    store = FAISS.load_local(
        str(index_path),
        embeddings=None,
        allow_dangerous_deserialization=True,
    )
    documents = list(store.docstore._dict.values())

    print(f"Dossier : {index_path}")
    print(f"Vecteurs FAISS : {store.index.ntotal}")
    print(f"Documents : {len(documents)}")
    print(f"Dimension des vecteurs : {store.index.d}")

    if not documents:
        print("Aucun document dans l'index.")
        return

    print(f"\nExemples ({min(limit, len(documents))}) :")
    for number, document in enumerate(documents[:limit], start=1):
        print(f"\n--- Document {number} ---")
        print("Métadonnées :", document.metadata)
        print("Contenu :")
        print(document.page_content)


def main():
    parser = argparse.ArgumentParser(
        description="Inspecte localement un index FAISS et ses documents."
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Nombre de documents à afficher (défaut : 5)."
    )
    parser.add_argument(
        "--index-path",
        default=str(FAISS_INDEX_PATH),
        help="Dossier contenant index.faiss et index.pkl.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit doit être positif ou nul.")
    read_index(index_path=Path(args.index_path), limit=args.limit)


if __name__ == "__main__":
    main()
