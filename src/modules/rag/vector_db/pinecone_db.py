from typing import List
import logging

from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.index import Index as PineconeIndex
import numpy as np

from .base import VectorDatabaseBase
from core.entities.types import DocumentChunk, EmbeddingArray, RagDocument
from common.utils import require_env_var

logger = logging.getLogger(__name__)


class PineconeDatabase(VectorDatabaseBase):
    """
    Vector database using Pinecone for similarity search and document storage.
    
    This class provides methods to store document embeddings and perform similarity searches
    using Pinecone's serverless vector database.

    Args:
        index_name (str): Name of the Pinecone index
        dimension (int): Vector dimension for embeddings
        metric (str): Distance metric for similarity search (e.g., 'cosine', 'euclidean')
        cloud (str): Cloud provider (e.g., 'aws', 'gcp', 'azure')
        region (str): Cloud region (e.g., 'us-east-1', 'us-west1')
     
    Raises:
        ValueError: If configuration is invalid
    """
    def __init__(self, 
                 index_name: str,
                 dimension: int = 768,
                 metric: str = 'cosine',
                 cloud: str = 'aws',
                 region: str = 'us-east-1'):
        super().__init__()
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        try:
            self._pinecone = Pinecone(api_key=require_env_var('PINECONE_API_KEY'))
            logger.debug("Successfully connected to Pinecone")

            self.index = self._create_index()
            logger.info(f"Successfully initialized index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize PineconeEngine: {e}", exc_info=True)
            raise

    def _create_index(self) -> PineconeIndex:
        """Create or retrieve existing Pinecone index."""
        try:
            # Check if index already exists
            existing_indexes = [index['name'] for index in self._pinecone.list_indexes()]
            logger.debug(f"Existing indexes: {existing_indexes}")

            if self.index_name not in existing_indexes:
                self._pinecone.create_index(
                    name=self.index_name, 
                    dimension=self.dimension, 
                    metric=self.metric, 
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region)
                )
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Using existing index: {self.index_name}")
            return self._pinecone.Index(self.index_name)
        
        except Exception as e:
            logger.error(f"Failed to create or retrieve index: {self.index_name}: {e}", exc_info=True)
            raise
        
    def add_documents(self, documents: List[RagDocument]):
        """Add document embeddings and metadata to the vector database."""
        try:
            # Prepare vectors for upsert
            logger.debug(f"Adding {len(documents)} documents to index")
            vectors = []
            for i, doc in enumerate(documents):
                for j, chunk in enumerate(doc.chunks):
                    vectors.append({
                        "id": f"{doc.id}_chunk_{j}",
                        "values": chunk.embeddings.tolist(),
                        "metadata": {
                            "doc_id": doc.id,
                            "text": chunk.content
                        }
                    })
                logger.debug(f"Prepared vector {i+1}/{len(documents)} with ID: {doc.id}")

            # Upsert vectors to Pinecone
            self.index.upsert(vectors)
            logger.debug(f"Successfully upserted {len(vectors)} vectors to index")

        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}", exc_info=True)
            raise

    # TODO: re-ranking after extracting documents
    def search(self, 
               query_embeddings: np.ndarray, 
               n_extracted_docs: int = 5) -> List[List[DocumentChunk]]:
        """
        Search for similar document chunks using query embeddings.

        Args:
            query_embeddings (np.ndarray): A single embedding or batch of embeddings, shaped (embedding_dim,) or (n_queries, embedding_dim).
            n_extracted_docs (int): Number of top similar document chunks to retrieve per query.

        Returns:
            List[List[DocumentChunk]]: A list of lists containing the top-k most similar document chunks for each query.

        Raises:
            ValueError: If input embedding dimensions are invalid.
            Exception: If search fails for any reason.
        """

        try:
            query_embeddings = self._normalize_embeddings(query_embeddings)
            logger.debug(f"Searching for {n_extracted_docs} similar document chunks for {len(query_embeddings)} queries")
            
            results = []
            for i, query in enumerate(query_embeddings):
                try:
                    # Perform similarity search
                    response = self.index.query(vector=query.tolist(),
                                                top_k=n_extracted_docs,
                                                include_metadata=True)
                    
                    # Extract matches and convert to DocumentChunk objects
                    matches = [DocumentChunk(document_id=doc['metadata']['doc_id'],
                                             content=doc['metadata']['text'],
                                             embeddings=None) 
                                             for doc in response['matches']]
                    
                    results.append(matches)
                    logger.debug(f"Query {i+1}/{len(query_embeddings)}: Found {len(matches)} matches")
                    
                except Exception as e:
                    logger.error(f"Failed to process query {i+1}/{len(query_embeddings)}", exc_info=True)
                    # Add empty result for failed query to maintain result structure
                    results.append([])
            
            logger.debug(f"Search completed. Retrieved results for {len(results)} queries")
            return results
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Search operation failed: {e}", exc_info=True)
            raise 

    def _normalize_embeddings(self, embeddings: EmbeddingArray) -> np.ndarray:
        """
        Validates and reshapes the input embeddings to ensure compatibility with the configured vector dimension.

        Returns:
            np.ndarray: A 2D numpy array where each row represents a single embedding vector.
        """
        embeddings = np.array(embeddings)
        
        # Validate embedding dimensions
        if embeddings.ndim == 1:
            # Single vector: ensure it matches expected dimensionality
            if embeddings.shape[0] != self.dimension:
                error_msg = f"Query embedding dimension {embeddings.shape[0]} doesn't match config dimension {self.dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            embeddings = embeddings.reshape(1, -1)  # Convert to 2D shape (1, dim) for consistency
        elif embeddings.ndim == 2:
            # Batch of vectors: validate second dimension matches configured dimension
            if embeddings.shape[1] != self.dimension:
                error_msg = f"Query embedding dimension {embeddings.shape[1]} doesn't match config dimension {self.dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Query embeddings must be 1D or 2D, got {embeddings.ndim}D"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return embeddings