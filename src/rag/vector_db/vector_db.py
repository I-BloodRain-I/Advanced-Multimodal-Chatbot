from typing import List, Dict, Any, Optional, Union

import numpy as np

from config import Config
from chatbot.types import RagDocument
from .engines import PineconeEngine

class VectorDatabase:
    """
    Wrapper class for a vector database engine used in retrieval-augmented generation (RAG) systems.
    
    This class encapsulates a specific vector search engine and allows configuration of how many
    documents should be retrieved during a query operation.
    
    Args:
        engine (Union[PineconeEngine]): The vector database engine instance (e.g., Pinecone) to be used.
        n_extracted_docs (Optional[int]): Number of documents to retrieve. If not provided, a default
                                          is fetched from the global configuration.
    """
    def __init__(self, 
                 engine: Union[PineconeEngine], 
                 n_extracted_docs: Optional[int] = None):
        self.engine = engine
        self.n_extracted_docs = n_extracted_docs or Config().get('rag.vector_db.n_extracted_docs')

    def add_raw_document(self,
                         embeddings: np.ndarray,
                         contents: List[str],
                         metadatas: List[Dict[str, Any]]):
        """Add raw document data by converting to RagDocument objects.
        
        Args:
            embeddings (np.ndarray): Document embeddings array of shape (n_docs, embedding_dim)
            contents (List[str]): List of document content strings
            metadatas (List[Dict[str, Any]]): List of metadata dictionaries

        Raises:
            ValueError: If input arrays have mismatched lengths
        """
        self.engine.add_raw_documents(embeddings=embeddings,
                                      contents=contents,
                                      metadatas=metadatas)

    def add_documents(self, 
                      embeddings: np.ndarray,
                      documents: List[RagDocument]):
        """Add document embeddings and metadata to the vector database.
        
        Args:
            embeddings (np.ndarray): Document embeddings array of shape (n_docs, embedding_dim)
            documents (List[RagDocument]): List of RagDocument objects with content and metadata

        Raises:
            ValueError: If embeddings and documents length mismatch
        """
        self.engine.add_documents(embeddings=embeddings,
                                  documents=documents)
        
    def search(self, query_embeddings: np.ndarray) -> List[List[RagDocument]]:
        """Search for similar documents using query embeddings.
        
        Args:
            query_embeddings (np.ndarray): Query embedding(s) of shape (n_queries, embedding_dim) or (embedding_dim,)
            
        Returns:
            List[List[RagDocument]]: List of lists containing top-k RagDocument objects for each query
            
        Raises:
            ValueError: If query embeddings have invalid dimensions
        """

        return self.engine.search(query_embeddings=query_embeddings,
                                  n_extracted_docs=self.n_extracted_docs)