from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from common.types import RagDocument

class VectorDatabaseEngine(ABC):
    """
    Abstract base class for a vector database engine.

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
    def add_raw_documents(self, 
                          embeddings: np.ndarray,
                          contents: List[str],
                          metadatas: List[Dict[str, Any]]):
        """Adds raw documents to the index using separate embeddings, contents, and metadata."""
        pass

    @abstractmethod
    def add_documents(self, 
                      embeddings: np.ndarray,
                      documents: List[RagDocument]):
        """Adds documents to the index using RagDocument instances."""
        pass

    @abstractmethod
    def search(self, 
               query_embeddings: np.ndarray, 
               n_extracted_docs: int = 5) -> List[List[RagDocument]]:
        """Performs a similarity search using the given query embeddings."""
        pass
