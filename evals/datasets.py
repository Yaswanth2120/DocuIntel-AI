"""Fixed-source public datasets with provenance and bounded downloads."""

import hashlib
import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import requests

ROOT = Path("data")
SOURCES = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "squad": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def download(name):
    if name not in SOURCES:
        raise ValueError("Unsupported dataset")
    ROOT.mkdir(exist_ok=True)
    path = ROOT / ("scifact.zip" if name == "scifact" else "squad-dev-v1.1.json")
    if not path.exists():
        temporary = path.with_suffix(".part")
        with requests.get(SOURCES[name], timeout=(10, 120), stream=True) as response:
            response.raise_for_status()
            total = 0
            with temporary.open("wb") as handle:
                for block in response.iter_content(65536):
                    total += len(block)
                    if total > 100_000_000:
                        raise ValueError("Dataset too large")
                    handle.write(block)
        temporary.replace(path)
    return path, {
        "dataset": name,
        "url": SOURCES[name],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-SA 4.0",
        "reference": "https://github.com/beir-cellar/beir"
        if name == "scifact"
        else "https://rajpurkar.github.io/SQuAD-explorer/",
    }


def load(name):
    path, manifest = download(name)
    if name == "scifact":
        with zipfile.ZipFile(path) as archive:
            corpus = [json.loads(line) for line in archive.read("scifact/corpus.jsonl").decode().splitlines()]
            queries = {
                str(row["_id"]): row["text"]
                for row in [
                    json.loads(line) for line in archive.read("scifact/queries.jsonl").decode().splitlines()
                ]
            }
            qrels = {}
            for line in archive.read("scifact/qrels/test.tsv").decode().splitlines()[1:]:
                query, document, relevance = line.split("\t")
                if int(relevance) > 0:
                    qrels.setdefault(query, []).append(document)
        documents = {str(row["_id"]): row.get("title", "") + "\n\n" + row["text"] for row in corpus}
        cases = [
            {"id": query, "question": queries[query], "sources": sources}
            for query, sources in sorted(qrels.items())
        ]
    else:
        source = json.loads(path.read_text())
        documents = {}
        cases = []
        for article in source["data"]:
            for paragraph in article["paragraphs"]:
                identifier = hashlib.sha256(paragraph["context"].encode()).hexdigest()[:24]
                documents[identifier] = paragraph["context"]
                for question in paragraph["qas"]:
                    cases.append(
                        {
                            "id": question["id"],
                            "question": question["question"],
                            "sources": [identifier],
                            "answers": [a["text"] for a in question["answers"]],
                            "topic": article["title"],
                        }
                    )
    manifest.update(documents=len(documents), queries=len(cases))
    return documents, cases, manifest


if __name__ == "__main__":
    for name in SOURCES:
        _, _, manifest = load(name)
        print(json.dumps(manifest))
