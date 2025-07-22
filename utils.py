import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_document(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext == '.pdf':
        doc = fitz.open(filepath)
        return " ".join([page.get_text() for page in doc])
    elif ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Formato no soportado (.pdf o .txt)")

def chunk_text(text, max_tokens=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        current_tokens = len(current_chunk.split())
        sentence_tokens = len(sentence.split())

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def vectorize_chunks(chunks):
    return model.encode(chunks)

def init_db():
    conn = sqlite3.connect("storage.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            vector BLOB
        )
    """)
    conn.commit()
    conn.close()

def store_chunks(chunks, vectors):
    conn = sqlite3.connect("storage.db")
    cursor = conn.cursor()
    for text, vector in zip(chunks, vectors):
        cursor.execute("INSERT INTO chunks (text, vector) VALUES (?, ?)", (text, vector.tobytes()))
    conn.commit()
    conn.close()

def find_most_similar_chunk(question):
    conn = sqlite3.connect("storage.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, vector FROM chunks")
    results = cursor.fetchall()
    conn.close()

    if not results:
        return "No hay datos vectorizados."

    texts = []
    vectors = []

    for _, text, vector_blob in results:
        texts.append(text)
        vectors.append(np.frombuffer(vector_blob, dtype=np.float32))

    question_vector = model.encode([question])
    similarities = cosine_similarity([question_vector[0]], vectors)[0]
    best_index = int(np.argmax(similarities))

    return texts[best_index]