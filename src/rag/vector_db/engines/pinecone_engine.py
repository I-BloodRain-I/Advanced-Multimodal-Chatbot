from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
import logging

from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.index import Index as PineconeIndex
import numpy as np

from .base import VectorDatabaseEngine
from config import Config
from common.types import RagDocument
from utils.validation import require_env_var

logger = logging.getLogger(__name__)

@dataclass
class PineconeConfig:
    """Configuration class for Pinecone vector database settings.

    Attributes:
        index_name (str): Name of the Pinecone index
        dimension (int): Vector dimension for embeddings
        metric (str): Distance metric for similarity search (e.g., 'cosine', 'euclidean')
        cloud (str): Cloud provider (e.g., 'aws', 'gcp', 'azure')
        region (str): Cloud region (e.g., 'us-east-1', 'us-west1')
    """

    index_name: str
    dimension:  int
    metric:     str
    cloud:      str
    region:     str

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **overrides) -> 'PineconeConfig':
        """Create PineconeConfig from configuration dictionary with optional overrides.
        
        Args:
            cfg (Dict[str, Any]): Configuration dictionary
            **overrides: Optional parameter overrides
            
        Returns:
            PineconeConfig: Configured PineconeConfig instance
        """
        try:
            return cls(
                index_name=overrides.get('index_name', cfg.get('index_name')),
                dimension=overrides.get('dimension', cfg.get('dim')),
                metric=overrides.get('metric', cfg.get('metric')),
                cloud=overrides.get('cloud', cfg.get('cloud')),
                region=overrides.get('region', cfg.get('region'))
            )
        except Exception as e:
            logger.error(f"Failed to create PineconeConfig from configuration: {e}", exc_info=True)
            raise

class PineconeEngine(VectorDatabaseEngine):
    """Vector database engine using Pinecone for similarity search and document storage.
    
    This class provides methods to store document embeddings and perform similarity searches
    using Pinecone's serverless vector database.

    Args:
        config (Optional[PineconeConfig]): Pinecone configuration. If None, loads from Config
        
    Raises:
        ValueError: If configuration is invalid
    """
    def __init__(self, config: Optional[PineconeConfig] = None):
        super().__init__()
        try:
            # Check if config in [None, PineconeConfig]
            if config is None:
                self._config = PineconeConfig.from_config(Config().get('rag.vector_db.pinecone'))
            elif isinstance(config, PineconeConfig):
                self._config = config
            else:
                logger.error(f"Expected PineconeConfig but got: {type(config)}")
                raise 

            self._pinecone = Pinecone(api_key=require_env_var('PINECONE_API_KEY'))
            logger.debug("Successfully connected to Pinecone")

            self.index = self._create_index()
            logger.info(f"Successfully initialized index: {self._config.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize PineconeEngine: {e}", exc_info=True)
            raise

    def _create_index(self) -> PineconeIndex:
        """Create or retrieve existing Pinecone index."""
        try:
            # Check if index already exists
            existing_indexes = [index['name'] for index in self._pinecone.list_indexes()]
            logger.debug(f"Existing indexes: {existing_indexes}")

            if self._config.index_name not in existing_indexes:
                self._pinecone.create_index(
                    name=self._config.index_name, 
                    dimension=self._config.dimension, 
                    metric=self._config.metric, 
                    spec=ServerlessSpec(cloud=self._config.cloud, region=self._config.region)
                )
                logger.info(f"Successfully created index: {self._config.index_name}")
            else:
                logger.info(f"Using existing index: {self._config.index_name}")
            return self._pinecone.Index(self._config.index_name)
        
        except Exception as e:
            logger.error(f"Failed to create or retrieve index: {self._config.index_name}: {e}", exc_info=True)
            raise
        
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
        try:
            embeddings = np.array(embeddings)
            
            # Validate input lengths
            if len(embeddings) != len(documents):
                error_msg = f"Length mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate embedding dimensions
            if embeddings.shape[1] != self._config.dimension:
                error_msg = f"Embedding dimension {embeddings.shape[1]} doesn't match config dimension {self._config.dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"Adding {len(documents)} documents to index")

            # Prepare vectors for upsert
            vectors = []
            for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
                vector_id = str(uuid.uuid4())
                vectors.append({
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "content": doc.content,
                        **doc.metadata
                    }
                })
                logger.debug(f"Prepared vector {i+1}/{len(documents)} with ID: {vector_id}")

            # Upsert vectors to Pinecone
            self.index.upsert(vectors)
            logger.debug(f"Successfully upserted {len(vectors)} vectors to index")

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}", exc_info=True)
            raise

    def add_raw_documents(self,
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
        try:
            embeddings = np.array(embeddings)

            # Validate input lengths
            if not (len(embeddings) == len(contents) == len(metadatas)):
                error_msg = f"Length mismatch: {len(embeddings)} embeddings, {len(contents)} contents, {len(metadatas)} metadatas"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Converting {len(contents)} raw documents to RagDocument objects")
            
            # Convert to RagDocument objects
            documents = [RagDocument(content=content, metadata=metadata)
                        for content, metadata in zip(contents, metadatas)]
            
            # Use existing add_documents method
            self.add_documents(embeddings, documents)
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Failed to add raw documents: {e}", exc_info=True)
            raise

    def search(self, 
               query_embeddings: np.ndarray, 
               n_extracted_docs: int = 5) -> List[List[RagDocument]]:
        try:
            query_embeddings = np.array(query_embeddings)
            
            # Validate embedding dimensions
            if query_embeddings.ndim == 1:
                if query_embeddings.shape[0] != self._config.dimension:
                    error_msg = f"Query embedding dimension {query_embeddings.shape[0]} doesn't match config dimension {self._config.dimension}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                query_embeddings = query_embeddings.reshape(1, -1)
            elif query_embeddings.ndim == 2:
                if query_embeddings.shape[1] != self._config.dimension:
                    error_msg = f"Query embedding dimension {query_embeddings.shape[1]} doesn't match config dimension {self._config.dimension}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                error_msg = f"Query embeddings must be 1D or 2D, got {query_embeddings.ndim}D"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Searching for {n_extracted_docs} similar documents for {len(query_embeddings)} queries")
            
            results = []
            for i, query in enumerate(query_embeddings):
                try:
                    # Perform similarity search
                    response = self.index.query(
                        vector=query.tolist(),
                        top_k=n_extracted_docs,
                        include_metadata=True
                    )
                    
                    # Extract matches and convert to RagDocument objects
                    matches = [
                        RagDocument(
                            content=doc['metadata'].pop('content'),
                            metadata=doc['metadata']
                        ) for doc in response['matches']
                    ]
                    
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