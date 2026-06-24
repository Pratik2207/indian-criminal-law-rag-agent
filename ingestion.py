import os
import glob
import uuid
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

class LegalDocumentIngestor:
    def __init__(self, knowledge_dir, collection_name="indian_law"):
        self.knowledge_dir = knowledge_dir
        self.collection_name = collection_name
        self.client = QdrantClient(path="./qdrant_db")
        self.md = MarkItDown()
        self.chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _extract_text(self, file_path):
        print(f"Extracting text from {file_path}...")
        try:
            result = self.md.convert(file_path)
            return result.text_content
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    def _create_chunks(self, raw_text):
        print("Creating semantic chunks...")
        return self.chunker.chunk(raw_text)

    def setup_collection(self):
        print(f"Setting up Qdrant collection: {self.collection_name}...")
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
            print(f"Collection {self.collection_name} created.")
        else:
            print(f"Collection {self.collection_name} already exists.")

    def ingest_documents(self):
        self.setup_collection()
        pdf_files = glob.glob(os.path.join(self.knowledge_dir, "*.pdf"))
        
        for pdf_file in pdf_files:
            raw_text = self._extract_text(pdf_file)
            if not raw_text:
                continue
                
            chunks = self._create_chunks(raw_text)
            texts = [chunk.text for chunk in chunks]
            
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} for {os.path.basename(pdf_file)}...")
                
                embeddings = list(self.embed_model.embed(batch_texts))
                
                points = []
                for j, (text, vector) in enumerate(zip(batch_texts, embeddings)):
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector.tolist(),
                        payload={
                            "source": os.path.basename(pdf_file),
                            "text": text
                        }
                    ))
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
        print("Ingestion complete.")

if __name__ == "__main__":
    ingestor = LegalDocumentIngestor(knowledge_dir="./knowledge")
    ingestor.ingest_documents()
