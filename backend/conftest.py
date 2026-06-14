"""pytest configuration.

Having this file at backend/ makes pytest insert backend/ on sys.path, so tests
can `import rag...` and `import app`. We also force the HuggingFace libraries into
offline mode: the embedding model is already cached from ingestion, so tests run
fast and don't hit the network.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.resolve()))
