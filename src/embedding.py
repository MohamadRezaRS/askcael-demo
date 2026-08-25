from sentence_transformers import SentenceTransformer
import numpy as np
import os
from src.config import USE_OFFLINE_MODEL, GEMINI_API_KEY
from google import genai

_offline_model = None

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
            _offline_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
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
            _offline_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        return _offline_model.encode(texts).tolist()
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.embed_content(
            model='models/gemini-embedding-001',
            contents=texts,
        )
        return [_truncate_and_normalize(emb.values) for emb in response.embeddings]
