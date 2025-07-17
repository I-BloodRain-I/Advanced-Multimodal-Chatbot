from abc import ABC, abstractmethod
from typing import List
from core.entities.types import EmbeddingArray, RagDocument, DocumentChunk

class VectorDatabaseBase(ABC):
    """
    Abstract base class for a vector database.

    This interface defines the core methods required for managing a vector database,
    including index creation, document addition, and similarity-based search.
    """

    @abstractmethod
    def _create_index(self):
        """
        Initializes or creates the underlying index structure for the vector database.

        This method should be called before adding or searching documents,
        and implemented by subclasses to define the indexing strategy.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[RagDocument]):
        """
        Add document embeddings and metadata to the vector database.
        
        Args:
            documents (List[RagDocument]): List of `RagDocument` objects that consist of `DocumentChunk` objects
        """
        pass

    @abstractmethod
    def search(self, 
               query_embeddings: EmbeddingArray, 
               n_extracted_docs: int = 5) -> List[List[DocumentChunk]]:
        """
        Search for similar document chunks using query embeddings.

        Args:
            query_embeddings (np.ndarray): A single embedding or batch of embeddings, shaped (embedding_dim,) or (n_queries, embedding_dim).
            n_extracted_docs (int): Number of top similar document chunks to retrieve per query.

        Returns:
            List[List[DocumentChunk]]: A list of lists containing the top-k most similar document chunks for each query.
        """
        pass
