from qdrant_client import QdrantClient

def check_qdrant():
    try:
        client = QdrantClient(path='./qdrant_db')
        collection_info = client.get_collection('indian_law')
        print(f"Collection 'indian_law' has {collection_info.points_count} points.")
        client.close()
    except Exception as e:
        print(f"Error checking Qdrant: {e}")

if __name__ == "__main__":
    check_qdrant()
