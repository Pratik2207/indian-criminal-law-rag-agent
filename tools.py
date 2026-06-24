from crewai_tools import tool
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# Load once (global objects)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(path="./qdrant_db")

COLLECTION_NAME = "indian_law"

@tool("IndianLawDocumentSearch")
def document_search_tool(query: str):
    """
    Searches Indian criminal law documents (IPC, CrPC, BNS, BNSS)
    stored in a local Qdrant vector database and returns
    the most relevant legal sections for the given query.
    """

    query_vector = embed_model.encode(query).tolist()

    try:
        # qdrant-client >= 1.10
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[],
            query=query_vector,
            limit=5,
            search_params=SearchParams(hnsw_ef=128),
        ).points
    except Exception:
        # Backward compatibility for older qdrant-client versions
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5,
            search_params=SearchParams(hnsw_ef=128),
        )

    if not results:
        return "No relevant legal sections found."

    responses = []
    for hit in results:
        payload = hit.payload or {}
        source = payload.get("source", "Unknown")
        text = payload.get("text", "")
        responses.append(f"Source: {source}\n{text}")

    return "\n\n---\n\n".join(responses)
