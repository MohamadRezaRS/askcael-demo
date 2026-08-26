import sys
import json
import warnings
warnings.filterwarnings("ignore")
from src.chains import (
    build_classifier_chain, 
    build_hyde_chain, 
    build_generation_chain
)
from src.database import get_connection, vector_search
from src.embedding import get_embedding
from src.config import USE_OFFLINE_MODEL, RETRIEVAL_TOP_N, FINAL_TOP_N

FALLBACK_OFF_TOPIC = "I'm a movie recommendation assistant. I can't help with that request."
FALLBACK_UNRECOGNIZED = "I'm not sure how to handle that phrasing. Could you rephrase your movie request?"

def get_all_titles():
    conn = get_connection()
    cursor = conn.cursor()
    table_name = 'offline' if USE_OFFLINE_MODEL else 'online'
    cursor.execute(f"SELECT title FROM {table_name}")
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles

def get_movie_summaries(titles):
    if not titles:
        return ""
    conn = get_connection()
    cursor = conn.cursor()
    table_name = 'offline' if USE_OFFLINE_MODEL else 'online'
    placeholders = ','.join(['?'] * len(titles))
    query = f"SELECT title, summary FROM {table_name} WHERE title IN ({placeholders})"
    cursor.execute(query, titles)
    rows = cursor.fetchall()
    conn.close()
    return "\n\n".join([f"Title: {r[0]}\nSummary: {r[1]}" for r in rows])

def get_stored_vector(title):
    conn = get_connection()
    cursor = conn.cursor()
    table_name = 'offline' if USE_OFFLINE_MODEL else 'online'
    cursor.execute(f"SELECT CAST(embedding AS NVARCHAR(MAX)) FROM {table_name} WHERE title = ?", (title,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def process_query(query: str) -> str:
    classifier_chain = build_classifier_chain()
    hyde_chain = build_hyde_chain()
    gen_chain = build_generation_chain()
    
    titles = get_all_titles()
    
    try:
        classification = classifier_chain.invoke({
            "titles": ", ".join(titles),
            "query": query
        })
    except Exception:
        return FALLBACK_UNRECOGNIZED
        
    if not getattr(classification, 'is_on_topic', False):
        return FALLBACK_OFF_TOPIC
        
    case = getattr(classification, 'query_case', None)
    if case is None or case not in range(1, 7):
        return FALLBACK_UNRECOGNIZED
        
    anchor_vector = None
    
    if case in [1, 2]:
        hyde_input = classification.cleaned_text or query
        if case == 2:
            vector_input_text = hyde_chain.invoke({
                "cleaned_text": hyde_input,
                "movie_summaries": "",
                "constraint_text": ""
            })
            anchor_vector = get_embedding(vector_input_text)
        else:
            anchor_vector = get_embedding(hyde_input)
            
    elif case == 3:
        if classification.referenced_titles:
            anchor_vector = get_stored_vector(classification.referenced_titles[0])
        if anchor_vector is None:
            return FALLBACK_UNRECOGNIZED
            
    elif case in [4, 5, 6]:
        if classification.referenced_titles:
            summaries = get_movie_summaries(classification.referenced_titles)
            vector_input_text = hyde_chain.invoke({
                "cleaned_text": "",
                "movie_summaries": summaries,
                "constraint_text": classification.constraint_text or ""
            })
            anchor_vector = get_embedding(vector_input_text)
        else:
            return FALLBACK_UNRECOGNIZED
            
    if anchor_vector is None:
        return FALLBACK_UNRECOGNIZED

    table_name = 'offline' if USE_OFFLINE_MODEL else 'online'
    candidates = vector_search(table_name, anchor_vector, top_n=RETRIEVAL_TOP_N)
    
    excluded_titles = {t.lower() for t in (classification.referenced_titles or [])}
    filtered_candidates = [c for c in candidates if c['title'].lower() not in excluded_titles]
    
    if not filtered_candidates:
        return "No matching candidates found after exclusions."
        
    candidates_text = "\n\n".join([f"Title: {c['title']}\nSummary: {c['short_summary']}" for c in filtered_candidates])
    
    response = gen_chain.invoke({
        "query": query,
        "constraint_text": classification.constraint_text or "",
        "candidates": candidates_text
    })
    
    return response

def main():
    while True:
        try:
            query = input("\nEnter your movie request (or 'quit'): ")
            if query.strip().lower() in ['quit', 'exit']:
                break
            if not query.strip():
                continue
                
            word_count = len(query.strip().split())
            if word_count > 300:
                print("\nError: Your request is too long. Please keep it under 300 words.\n")
                continue
            
            response = process_query(query)
            print(f"\n{response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
