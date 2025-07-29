"""
RAG (Retrieval-Augmented Generation) module for the TensorAlix Agent AI system.

This module provides document indexing and retrieval capabilities to enhance
language model responses with relevant contextual information. It supports
multiple vector database backends and handles embedding-based similarity search.

Classes:
    RAG: Core retrieval-augmented generation pipeline
    VectorDatabaseBase: Abstract base class for vector databases
    PineconeDatabase: Pinecone vector database implementation
"""

from .rag import RAG
from .vector_db import PineconeDatabase, VectorDatabaseBase