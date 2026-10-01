from flask import Flask, jsonify, request
from flask_cors import CORS

from src.vector_store import get_collection
from src.rag_pipeline import answer_query


app = Flask(__name__)

# Allow requests from Streamlit or other frontend applications
CORS(app)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "success": True,
        "message": "Multimodal RAG API is running",
        "status": "healthy"
    })


# ============================================================
# RAG QUESTION API
# ============================================================

@app.route("/ask", methods=["POST"])
def ask_question():

    try:

        # ----------------------------------------------------
        # Get JSON request body
        # ----------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must contain JSON data."
            }), 400

        # ----------------------------------------------------
        # Get query
        # ----------------------------------------------------

        query = data.get("query")

        if not query:
            return jsonify({
                "success": False,
                "error": "The 'query' field is required."
            }), 400

        if not isinstance(query, str):
            return jsonify({
                "success": False,
                "error": "The 'query' field must be a string."
            }), 400

        query = query.strip()

        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty."
            }), 400

        # ----------------------------------------------------
        # Get retrieval settings
        # ----------------------------------------------------

        pdf_top_k = data.get("pdf_top_k", 5)
        image_top_k = data.get("image_top_k", 1)

        try:

            pdf_top_k = int(pdf_top_k)
            image_top_k = int(image_top_k)

        except (TypeError, ValueError):

            return jsonify({
                "success": False,
                "error": "pdf_top_k and image_top_k must be integers."
            }), 400

        if pdf_top_k < 1 or image_top_k < 1:

            return jsonify({
                "success": False,
                "error": "pdf_top_k and image_top_k must be greater than 0."
            }), 400

        # ----------------------------------------------------
        # Get ChromaDB collection
        # ----------------------------------------------------

        collection = get_collection()

        # ----------------------------------------------------
        # Run existing RAG pipeline
        # ----------------------------------------------------

        result = answer_query(
            collection=collection,
            query=query,
            pdf_top_k=pdf_top_k,
            image_top_k=image_top_k
        )

        # ----------------------------------------------------
        # Return JSON response
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "query": query,
            "answer": result.get("answer", ""),
            "pdf_results": result.get("pdf_results", []),
            "image_results": result.get("image_results", []),
            "image_sources": result.get("image_sources", [])
        })

    except Exception as error:

        print("\nAPI ERROR:")
        print(error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )