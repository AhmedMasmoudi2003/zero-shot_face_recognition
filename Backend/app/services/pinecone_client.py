from pinecone import Pinecone
from app.core.config import settings

class PineconeClient:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=512,
                metric="cosine"
            )

    def upsert_face(self, person_id: str, embedding: list, name: str):
        vector = {
            "id": person_id,
            "values": embedding,
            "metadata": {"name": name}
        }
        return self.index.upsert(vectors=[vector])

    def query_face(self, embedding: list, top_k: int = settings.TOP_K):
        return self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

    def delete_vector(self, vector_id: str):
        self.index.delete(ids=[vector_id])

# Singleton instance
pinecone_client = PineconeClient()
