"""
Vector store loader for Lambda.

On first call, downloads chroma.tar.gz from S3 into /tmp,
extracts it, and loads a ChromaDB in-memory client.
Subsequent calls reuse the cached client (warm Lambda).
"""

import json
import os
import tarfile
import tempfile
from pathlib import Path

import boto3
import chromadb

BUCKET = os.environ.get("KNOWLEDGE_BUCKET", "")
S3_KEY = "knowledge/chroma.tar.gz"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EXTRACT_PATH = Path("/tmp/chroma_store")

_collection = None  # module-level cache


def _download_and_extract():
    s3 = boto3.client("s3")
    tar_path = "/tmp/chroma.tar.gz"
    s3.download_file(BUCKET, S3_KEY, tar_path)
    EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(str(EXTRACT_PATH))


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    chroma_dir = EXTRACT_PATH / "chroma"
    if not chroma_dir.exists():
        _download_and_extract()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    _collection = client.get_collection("study_knowledge")
    return _collection


def _embed(text: str) -> list[float]:
    bedrock = boto3.client("bedrock-runtime")
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def search(query: str, n_results: int = 5) -> list[dict]:
    """
    Embed query and return top-n matching chunks with metadata.
    Returns list of {"text": ..., "source": ..., "title": ..., "distance": ...}
    """
    collection = _get_collection()
    embedding = _embed(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": round(dist, 4),
        })
    return hits
