"""
One-time ingestion script — run locally before deploying.

Reads PDFs from documents/, chunks them, embeds via Bedrock Titan,
saves embeddings as a gzipped JSON file to S3.

Usage:
    pip install pypdf boto3
    python ingest.py

Env vars needed (or set in .env):
    AWS_DEFAULT_REGION   (e.g. us-east-1)
    KNOWLEDGE_BUCKET     (S3 bucket name — get from SAM deploy output)
"""

import gzip
import json
import os
from pathlib import Path

import boto3
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.environ.get("KNOWLEDGE_BUCKET")
S3_KEY = "knowledge/embeddings.json.gz"
DOCS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 400      # words per chunk
CHUNK_OVERLAP = 50    # words overlap between chunks
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

PAPER_METADATA = {
    "attention_is_all_you_need.pdf": {
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al., 2017",
    },
    "formal_algorithms.pdf": {
        "title": "Formal Algorithms for Transformers",
        "authors": "Phuong & Hutter, DeepMind",
    },
}


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(text: str, bedrock) -> list[float]:
    body = json.dumps({"inputText": text[:8000]})  # Titan v2 limit
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def main():
    if not BUCKET:
        raise ValueError("Set KNOWLEDGE_BUCKET env var (get it from SAM deploy output)")

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    records = []

    for pdf_file in sorted(DOCS_DIR.glob("*.pdf")):
        meta = PAPER_METADATA.get(pdf_file.name, {"title": pdf_file.stem, "authors": ""})
        print(f"\nProcessing: {pdf_file.name}")
        text = extract_text(pdf_file)
        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            embedding = embed(chunk, bedrock)
            records.append({
                "id": f"{pdf_file.stem}__chunk{i}",
                "text": chunk,
                "embedding": embedding,
                "source": pdf_file.name,
                "title": meta["title"],
                "authors": meta["authors"],
                "chunk_index": i,
            })
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i+1}/{len(chunks)}")

    print(f"\nTotal chunks: {len(records)}")
    print(f"Uploading to s3://{BUCKET}/{S3_KEY} ...")

    payload = gzip.compress(json.dumps(records).encode())
    s3.put_object(Bucket=BUCKET, Key=S3_KEY, Body=payload, ContentEncoding="gzip")

    print("Done! Knowledge base uploaded to S3.")


if __name__ == "__main__":
    main()
