import requests


FLASK_API_URL = "http://127.0.0.1:5000"


def ask_rag_api(
    query,
    pdf_top_k=5,
    image_top_k=1
):
    """
    Send a user query to the Flask RAG API.

    Parameters
    ----------
    query : str
        User's natural-language question.

    pdf_top_k : int
        Number of PDF results to retrieve.

    image_top_k : int
        Number of image results to retrieve.

    Returns
    -------
    dict
        JSON response returned by the Flask API.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    payload = {
        "query": query.strip(),
        "pdf_top_k": pdf_top_k,
        "image_top_k": image_top_k
    }

    try:
        response = requests.post(
            f"{FLASK_API_URL}/ask",
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(
                result.get(
                    "error",
                    "The RAG API returned an error."
                )
            )

        return result

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to the Flask API. "
            "Make sure the Flask server is running on port 5000."
        ) from error

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "The Flask API request timed out. "
            "The RAG pipeline may still be processing."
        ) from error

    except requests.exceptions.HTTPError as error:
        try:
            error_details = response.json().get(
                "error",
                "Unknown API error."
            )
        except Exception:
            error_details = response.text

        raise RuntimeError(
            f"Flask API returned an HTTP error: {error_details}"
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Could not communicate with the Flask API: {error}"
        ) from error