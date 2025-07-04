import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple


class VectorStore:
    """
    A wrapper around FAISS to store, retrieve, and manage document embeddings,
    along with external metadata and original documents.
    """
    _FAISS_INDEX_FILENAME = "faiss_index.idx"
    _METADATA_FILENAME = "metadata.pkl"
    
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.dimension = dimension
        self.documents: List[str] = []
        self.metadata: List[Dict] = []


    def add_documents(self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict]
    ):
        """
        Adds embeddings, documents, and associated metadata to the store.
        """
        assert len(embeddings) == len(documents) == len(metadatas), "Length mismatch"
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(documents)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, Dict]]:
        """
        Returns top-k matching documents based on the query embedding.
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append((self.documents[idx], self.metadata[idx]))
        return results

    def save(self, path: str):
        """
        Saves FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, f"{path}/{self._FAISS_INDEX_FILENAME}")
        with open(f"{path}/{self._METADATA_FILENAME}", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f)

    @classmethod
    def load(cls, path: str = './vector_db') -> "VectorStore":
        """
        Loads FAISS index and metadata from disk.
        """
        index = faiss.read_index(f"{path}/{cls._FAISS_INDEX_FILENAME}")
        with open(f"{path}/{cls._METADATA_FILENAME}", "rb") as f:
            data = pickle.load(f)

        store = cls(data["dimension"])
        store.index = index
        store.documents = data["documents"]
        store.metadata = data["metadata"]
        return store
