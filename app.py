from flask import Flask, request, jsonify
from utils import load_document, chunk_text, vectorize_chunks, store_chunks, init_db, find_most_similar_chunk

app = Flask(__name__)
init_db()

@app.route('/procesar', methods=['POST'])
def procesar():
    filepath = request.json.get('filepath')
    if not filepath:
        return jsonify({"error": "Se requiere filepath"}), 400

    try:
        text = load_document(filepath)
        chunks = chunk_text(text)
        vectors = vectorize_chunks(chunks)
        store_chunks(chunks, vectors.astype('float32'))
        return jsonify({"status": "procesado", "chunks": len(chunks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/preguntar', methods=['POST'])
def preguntar():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Se requiere una pregunta"}), 400
    answer = find_most_similar_chunk(question)
    return jsonify({"respuesta": answer})

if __name__ == '__main__':
    app.run(debug=True)