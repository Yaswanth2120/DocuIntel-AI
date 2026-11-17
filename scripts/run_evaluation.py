"""Run public RAG evaluation inside the configured Compose deployment.

Keeps database/cache ports private and uses only the restricted API database
credentials. Connection secrets are passed through a private temporary env file.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scifact", "squad"), default="scifact")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--generation-backend", choices=("local", "configured"), default="local")
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args()
    if args.queries < 1 or args.generate < 0:
        parser.error("queries must be positive and generate must be nonnegative")
    config = json.loads(
        subprocess.check_output(["docker", "compose", "config", "--format", "json"], cwd=ROOT, text=True)
    )
    api = config["services"]["api"]
    configured = api["environment"]
    allowed = {
        "DATABASE_URL",
        "REDIS_URL",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "OPENAI_TEMPERATURE",
        "MODEL_REASONING_EFFORT",
        "MAX_OUTPUT_TOKENS",
        "INPUT_USD_PER_MILLION",
        "OUTPUT_USD_PER_MILLION",
    }
    environment = {
        name: str(value)
        for name, value in configured.items()
        if value is not None and (name in allowed or name.startswith("RAG_"))
    }
    environment.update(
        HF_HOME="/models", MODEL_THREADS="2", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1"
    )
    environment["OPENAI_API_KEY"] = (
        configured.get("OPENAI_API_KEY", "") if args.generation_backend == "configured" else ""
    )
    if args.generation_backend == "configured" and not environment["OPENAI_API_KEY"]:
        parser.error("Configure the generation provider in .env before evaluating it")
    model_cache = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).resolve()
    data_cache = Path.home() / ".cache" / "docuintel-ai" / "data"
    for directory in (model_cache, data_cache, ROOT / "evals" / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--memory=1200m",
        "--cpus=2",
    ]
    for network in api["networks"]:
        command += ["--network", config["networks"][network]["name"]]
    command += [
        "-v",
        f"{ROOT / 'evals'}:/app/evals",
        "-v",
        f"{data_cache}:/app/data",
        "-v",
        f"{model_cache}:/models",
    ]
    with tempfile.TemporaryDirectory(prefix="docuintel-eval-") as temporary:
        connection_file = Path(temporary) / "connections.env"
        if any("\n" in value for value in environment.values()):
            raise ValueError("Evaluation environment values cannot contain newlines")
        connection_file.write_text("\n".join(f"{name}={value}" for name, value in environment.items()) + "\n")
        connection_file.chmod(0o600)
        command += [
            "--env-file",
            str(connection_file),
            api["image"],
            "python",
            "-u",
            "-m",
            "evals.public_benchmark",
            "--dataset",
            args.dataset,
            "--queries",
            str(args.queries),
            "--generate",
            str(args.generate),
            "--generation-backend",
            args.generation_backend,
        ]
        if args.generation_only:
            command.append("--generation-only")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
