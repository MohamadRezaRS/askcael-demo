import time
import warnings
warnings.filterwarnings("ignore")
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.database import get_connection

def add_short_summary_column():
    conn = get_connection()
    cursor = conn.cursor()
    for table in ['online', 'offline']:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD short_summary NVARCHAR(MAX)")
            conn.commit()
            print(f"Added short_summary column to {table} table.")
        except Exception:
            pass
    conn.close()

def generate_short_summary(summary, llm, chain):
    return chain.invoke({"summary": summary})

def main():
    add_short_summary_column()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title, summary, short_summary FROM online")
    rows = cursor.fetchall()
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following movie plot in exactly 50 words or less. Keep the core narrative and main characters. Do not include introductory phrases like 'This movie is about'."),
        ("human", "{summary}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    print(f"Found {len(rows)} movies. Starting summarization...", flush=True)
    
    for i, row in enumerate(rows):
        movie_id, title, summary, short_summary = row
        if short_summary:
            print(f"[{i+1}/{len(rows)}] Skipping {title} (already has short summary)", flush=True)
            continue
            
        print(f"[{i+1}/{len(rows)}] Summarizing {title}...", flush=True)
        try:
            new_short_summary = generate_short_summary(summary, llm, chain)
            
            cursor.execute("UPDATE online SET short_summary = ? WHERE title = ?", (new_short_summary, title))
            cursor.execute("UPDATE offline SET short_summary = ? WHERE title = ?", (new_short_summary, title))
            conn.commit()
            
            time.sleep(4.1)
        except Exception as e:
            print(f"Error summarizing {title}: {e}", flush=True)
            time.sleep(10)
            
    conn.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
