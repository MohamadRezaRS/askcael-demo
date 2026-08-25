from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from src.config import USE_OFFLINE_MODEL

_offline_model = None
_online_model = None

def get_embedding(text, use_offline=None):
    global _offline_model, _online_model
    if use_offline is None:
        use_offline = USE_OFFLINE_MODEL
        
    if use_offline:
        if _offline_model is None:
            _offline_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        return _offline_model.encode(text).tolist()
    else:
        if _online_model is None:
            _online_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return _online_model.embed_query(text)

def get_embeddings(texts, use_offline=None):
    global _offline_model, _online_model
    if use_offline is None:
        use_offline = USE_OFFLINE_MODEL
        
    if use_offline:
        if _offline_model is None:
            _offline_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        return _offline_model.encode(texts).tolist()
    else:
        if _online_model is None:
            _online_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return _online_model.embed_documents(texts)
