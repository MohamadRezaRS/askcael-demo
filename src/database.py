import pyodbc
import numpy as np
import json
from src.config import DB_SERVER, DB_NAME, USE_WINDOWS_AUTH, DB_USER, DB_PASSWORD

def get_connection():
    if USE_WINDOWS_AUTH:
        conn_str = f"Driver={{ODBC Driver 17 for SQL Server}};Server={DB_SERVER};Database={DB_NAME};Trusted_Connection=yes;"
    else:
        conn_str = f"Driver={{ODBC Driver 17 for SQL Server}};Server={DB_SERVER};Database={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD};"
    return pyodbc.connect(conn_str)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='online' and xtype='U')
        CREATE TABLE online (
            id INT IDENTITY(1,1) PRIMARY KEY,
            title NVARCHAR(255),
            summary NVARCHAR(MAX),
            word_count INT,
            embedding VECTOR(768)
        )
    """)
    
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='offline' and xtype='U')
        CREATE TABLE offline (
            id INT IDENTITY(1,1) PRIMARY KEY,
            title NVARCHAR(255),
            summary NVARCHAR(MAX),
            word_count INT,
            embedding VECTOR(768)
        )
    """)
    
    conn.commit()
    conn.close()

def insert_movie(table_name, title, summary, word_count, vector):
    conn = get_connection()
    cursor = conn.cursor()
    vector_str = json.dumps(vector)
    query = f"INSERT INTO {table_name} (title, summary, word_count, embedding) VALUES (?, ?, ?, CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(768)))"
    cursor.execute(query, (title, summary, word_count, vector_str))
    conn.commit()
    conn.close()

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def vector_search(table_name, query_vector, top_n=10):
    conn = get_connection()
    cursor = conn.cursor()
    query = f"SELECT title, summary, CAST(embedding AS NVARCHAR(MAX)) FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        title, summary, emb_str = row
        emb_arr = np.array(json.loads(emb_str))
        sim = cosine_similarity(query_vector, emb_arr)
        results.append({"similarity": sim, "title": title, "summary": summary})
        
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]
