"""
RAG (Retrieval-Augmented Generation) module for document indexing and contextual prompt enrichment.

This module handles:
- Storing structured documents in a vector database
- Retrieving similar documents using embeddings
- Injecting contextual information into messages for downstream use
"""

import logging
from typing import List

import numpy as np
from torch import Tensor

from core.entities.types import DocumentChunk, EmbeddingArray, Message, MessageHistory, RagDocument
from modules.rag.vector_db.base import VectorDatabaseBase

logger = logging.getLogger(__name__)


class RAG:
    """
    Retrieval-Augmented Generation pipeline for indexing documents and enriching prompts with context.

    Args:
        vector_db (VectorDatabaseBase): Backend vector database to store and query documents.
        n_extracted_docs (int): Number of similar documents to retrieve per query.
        prompt_format (str): Template for inserting context into prompts, must include '{context}' and '{prompt}'.
    """
    def __init__(self, 
                 vector_db: VectorDatabaseBase, 
                 n_extracted_docs: int = 5, 
                 prompt_format: str = '{context}\n{prompt}'):
        self.vector_db = vector_db
        self.n_extracted_docs = n_extracted_docs
        self.prompt_format = prompt_format

    def _validate_prompt_format(self):
        """
        Ensures the prompt_format contains required placeholders '{context}' and '{prompt}'.
        """
        if '{context}' not in self.prompt_format or '{prompt}' not in self.prompt_format:
            raise ValueError("prompt_format must include '{context}' and '{prompt}'")

    def add_documents(self, documents: List[RagDocument]):
        """
        Index pre-structured RagDocument into the vector database.

        Refer to :meth:`VectorDatabaseBase.add_documents` for details.
        """
        return self.vector_db.add_documents(documents)


    def extract_similar_docs(self, embeddings: EmbeddingArray) -> List[List[DocumentChunk]]:
        """
        Retrieves documents most similar to the input embeddings from the vector database.

        Args:
            embeddings (Union[List, np.ndarray, Tensor]): Query embeddings for document retrieval.

        Returns:
            List[List[DocumentChunk]]: A list of lists, each containing retrieved DocumentChunk per input embedding.
        """
        # Normalize embeddings to numpy format and perform search
        normalized = self._normalize_embeddings(embeddings)
        return self.vector_db.search(normalized, n_extracted_docs=self.n_extracted_docs)
    
    def add_context(self, messages_batch: List[MessageHistory], embeddings: EmbeddingArray):
        """
        Enriches the latest message in each conversation with relevant document context.

        Retrieves documents based on provided embeddings and inserts them into the message content
        using the configured prompt_format.

        Args:
            messages_batch (List[MessageHistory]): A batch of message histories.
            embeddings (EmbeddingArray): Query embeddings to retrieve relevant documents.
        """
        try:
            # Normalize the input embeddings to ensure consistency in similarity search
            embeddings = self._normalize_embeddings(embeddings)

            # Retrieve relevant documents for each embedding vector
            retrieved_docs_batch = self.extract_similar_docs(embeddings)

            for history, docs in zip(messages_batch, retrieved_docs_batch):
                if not history:
                    logger.warning("Empty conversation history encountered. Skipping.")
                    continue

                context = "\n".join([doc.content for doc in docs])

                # Replace the latest message content with context-injected version
                latest_message: Message = history[-1]
                history[-1] = Message(
                    role=latest_message.role,
                    content=self.prompt_format.format(
                        context=context, 
                        prompt=latest_message.content
                    )
                )
                
        except Exception as e:
            logger.error(
                f"Failed to inject context into message batch of size {len(messages_batch)}: {e}",
                exc_info=True
            )
            raise

    def _normalize_embeddings(self, embeddings: EmbeddingArray) -> np.ndarray:
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