from typing import List, Union, Optional

import numpy as np
from torch import Tensor

from config import Config
from chatbot.types import RagDocument
from .vector_db import VectorDatabase
from .vector_db.engines import PineconeEngine, PineconeConfig

class RAG:
    """
    Singleton class for handling Retrieval-Augmented Generation (RAG) operations.

    Responsible for initializing and managing a vector database engine,
    and for extracting similar documents based on input embeddings.

    Args:
        vector_db_engine_name (Optional[str]): Name of the vector DB engine (e.g., 'pinecone' or 'faiss').
        engine_config (Optional[Union[PineconeConfig]]): Configuration specific to the vector DB engine.
        n_extracted_docs (Optional[int]): Number of documents to extract per query.
    """
    _instance = None

    def __new__(cls):
        # Ensure only one instance of RAG is ever created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = Config()
        return cls._instance
    
    def __init__(self, 
                 vector_db_engine_name: Optional[str], 
                 engine_config: Optional[Union[PineconeConfig]] = None,
                 n_extracted_docs: Optional[int] = None):

        # Prevent reinitialization if vector_db is already set
        if hasattr(self, 'vector_db') and self.vector_db is not None:
            return

        self._config = Config()
        self.engine_config = engine_config
        self.n_extracted_docs = n_extracted_docs

        # Initialize vector database
        self.vector_db = self._load_vector_db(vector_db_engine_name)

    def _load_vector_db(self, engine_name: str) -> VectorDatabase:
        """
        Loads the vector database engine based on the specified name.

        Args:
            engine_name (str): The name of the vector DB engine to use.

        Returns:
            VectorDatabase: An initialized vector database instance.
        """
        if not engine_name:
            # Fallback to config-defined engine if none provided
            engine_name = self._config.get('rag.vector_db.engine')

        if engine_name == 'pinecone':
            # Initialize Pinecone engine; use provided or default config
            engine = PineconeEngine(self.engine_config)
            self.engine_config = self.engine_config or engine._config

        elif engine_name == 'faiss':
            # TODO: Implement FAISS engine configuration
            raise NotImplementedError("FAISS engine not yet implemented")
        else:
            raise ValueError(f"Unsupported engine: {engine_name}. Supported engines: ['pinecone', 'faiss']")
        
        return VectorDatabase(engine, self.n_extracted_docs)
    
    def extract_similar_docs(self, 
                             embeddings: Union[List[Union[List, float], float], np.ndarray, Tensor]) -> List[List[RagDocument]]:
        """
        Retrieves documents most similar to the input embeddings from the vector database.

        Args:
            embeddings (Union[List, np.ndarray, Tensor]): Query embeddings for document retrieval.

        Returns:
            List[List[RagDocument]]: A list of lists, each containing retrieved RagDocuments per input embedding.
        """
        # Normalize embeddings to numpy format and perform search
        return self.vector_db.search(self._normalize_embeddings(embeddings))
    
    def _normalize_embeddings(self, embeddings: Union[List[float], List[List[float]], np.ndarray, Tensor]) -> np.ndarray:
        """
        Converts input embeddings to a numpy array format for compatibility with the vector DB.
        """
        if isinstance(embeddings, Tensor):
            return embeddings.cpu().numpy()
        elif isinstance(embeddings, np.ndarray):
            return embeddings
        elif isinstance(embeddings, list):
            return np.array(embeddings)
        else:
            raise ValueError(f"Unsupported embeddings type: {type(embeddings)}")