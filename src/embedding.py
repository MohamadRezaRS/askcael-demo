import os
import sys
import io
import logging
import warnings
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from src.config import USE_OFFLINE_MODEL, GEMINI_API_KEY
from google import genai

_offline_model = None

def init_embedding_model():
    global _offline_model
    if USE_OFFLINE_MODEL and _offline_model is None:
        old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            _offline_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        finally:
            sys.stderr = old_stderr

def _truncate_and_normalize(vector, dim=768):
    arr = np.array(vector[:dim])
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

def get_embedding(text, use_offline=None):
    global _offline_model
    if use_offline is None:
        use_offline = USE_OFFLINE_MODEL
        
    if use_offline:
        if _offline_model is None:
            init_embedding_model()
        return _offline_model.encode(text).tolist()
    else:
        import time
        client = genai.Client(api_key=GEMINI_API_KEY)
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model='models/gemini-embedding-001',
                    contents=text,
                )
                return _truncate_and_normalize(response.embeddings[0].values)
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(2)

def get_embeddings(texts, use_offline=None):
    global _offline_model
    if use_offline is None:
        use_offline = USE_OFFLINE_MODEL
        
    if use_offline:
        if _offline_model is None:
            init_embedding_model()
        return _offline_model.encode(texts).tolist()
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.embed_content(
            model='models/gemini-embedding-001',
            contents=texts,
        )
        return [_truncate_and_normalize(emb.values) for emb in response.embeddings]

