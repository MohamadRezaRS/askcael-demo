import json
import os
import time
from src.database import init_db, insert_movie
from src.embedding import get_embedding
from src.config import USE_OFFLINE_MODEL

def main():
    init_db()
    
    table_name = 'offline' if USE_OFFLINE_MODEL else 'online'
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'movies.json')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        movies_data = json.load(f)
        
    total_movies = len(movies_data)
    current = 0
    
    for title, summary in movies_data.items():
        current += 1
        word_count = len(summary.split())
        vector = get_embedding(summary)
        insert_movie(table_name, title, summary, word_count, vector)
        print(f"[{current}/{total_movies}] Inserted {title} into {table_name} ({word_count} words)")
        
        if not USE_OFFLINE_MODEL:
            time.sleep(0.6)

if __name__ == "__main__":
    main()
