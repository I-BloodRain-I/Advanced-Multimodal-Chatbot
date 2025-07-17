import pickle
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

import faiss
import numpy as np

from config import Config
from common.types import RagDocument

@dataclass
class IndexConfig:
    dimension: int
    index_type: str
    flat_max_docs: int
    ivf_max_clusters: int
    ivf_max_probe: int
    ivf_scaling_factor: float
    ivf_max_docs: int
    ivfpq_max_clusters: int
    ivfpq_max_probe: int
    ivfpq_scaling_factor: float
    ivfpq_n_subvectors: int
    ivfpq_n_bits: int

    @classmethod
    def from_config(cls, db_cfg: Dict[str, Any], **overrides):
        flat_cfg  = db_cfg.get('flat_index_type')
        ivf_cfg   = db_cfg.get('inv_index_type')
        ivfpq_cfg = db_cfg.get('ivfpq_index_type')

        return cls(
            dimension=overrides.get('dimension', db_cfg.get('embed_dim')),
            index_type=overrides.get('index_type', db_cfg.get('index_type')),
            flat_max_docs=overrides.get('flat_max_docs', flat_cfg.get('flat_max_docs')),
            ivf_max_clusters=overrides.get('ivf_max_clusters', ivf_cfg.get('ivf_max_clusters')),
            ivf_max_probe=overrides.get('ivf_max_probe', ivf_cfg.get('ivf_max_probe')),
            ivf_scaling_factor=overrides.get('ivf_scaling_factor', ivf_cfg.get('ivf_scaling_factor')),
            ivf_max_docs=overrides.get('ivf_max_docs', ivf_cfg.get('ivf_max_docs')),
            ivfpq_max_clusters=overrides.get('ivfpq_max_clusters', ivfpq_cfg.get('ivfpq_max_clusters')),
            ivfpq_max_probe=overrides.get('ivfpq_max_probe', ivfpq_cfg.get('ivfpq_max_probe')),
            ivfpq_scaling_factor=overrides.get('ivfpq_scaling_factor', ivfpq_cfg.get('ivfpq_scaling_factor')),
            ivfpq_n_subvectors=overrides.get('ivfpq_n_subvectors', ivfpq_cfg.get('ivfpq_n_subvectors')),
            ivfpq_n_bits=overrides.get('ivfpq_n_bits', ivfpq_cfg.get('ivfpq_n_bits')),
        )

class VectorStore:
    """
    A wrapper around FAISS to store, retrieve, and manage document embeddings,
    along with external metadata and original documents.
    """
    _FAISS_INDEX_FILENAME = "faiss_index.idx"
    _METADATA_DB_FILENAME = "metadata.db"
    _CONFIG_FILENAME = "config.json"
    _EMBEDDINGS_FILENAME = "embeddings.npz"
    _config = Config()
    
    def __init__(self, 
                 dimension:            int = None,
                 index_type:           str = None,
                 device:               str = None,
                 flat_max_docs:        int = None,
                 ivf_max_clusters:     int = None,
                 ivf_max_probe:        int = None,
                 ivf_scaling_factor:   float = None,
                 ivf_max_docs:         int = None,
                 ivfpq_max_clusters:   int = None,
                 ivfpq_max_probe:      int = None,
                 ivfpq_scaling_factor: float = None,
                 ivfpq_n_subvectors:   int = None,
                 ivfpq_n_bits:         int = None):
        """index_type: "flat", "ivf", "ivfpq", "auto"""
        db_cfg = self._config.get('rag.vector_db')
        self._index_cfg = IndexConfig.from_config(
            db_cfg=db_cfg,
            dimension=dimension,
            index_type=index_type,
            flat_max_docs=flat_max_docs,
            ivf_max_clusters=ivf_max_clusters,
            ivf_max_probe=ivf_max_probe,
            ivf_scaling_factor=ivf_scaling_factor,
            ivf_max_docs=ivf_max_docs,
            ivfpq_max_clusters=ivfpq_max_clusters,
            ivfpq_max_probe=ivfpq_max_probe,
            ivfpq_scaling_factor=ivfpq_scaling_factor,
            ivfpq_n_subvectors=ivfpq_n_subvectors,
            ivfpq_n_bits=ivfpq_n_bits
        )

        self.index = self._create_index()

        self._device = device
        self._dir = db_cfg.get('dir')
        self._total_documents = 0
        self._documents_cache = {}
        self._cache_size = 1000

    def _create_index(self) -> faiss.Index:
        index_cfg = self._index_cfg
        index_type = index_cfg.index_type

        if index_type == "auto":
            # Автоматический выбор на основе размера и приоритета точности
            if self._total_documents < index_cfg.flat_max_docs:
                index_type = "flat"
            elif self._total_documents < index_cfg.ivf_max_docs:
                index_type = "ivf"
            else:
                index_type = "ivfpq"
        
        if index_type == "flat":
            # Максимальная точность, медленный поиск
            index = faiss.IndexFlatL2(index_cfg.dimension)
            
        elif index_type == "ivf":
            # Больше кластеров = выше точность, медленнее поиск
            nlist = min(int(self._total_documents ** index_cfg.ivf_scaling_factor), index_cfg.ivf_max_clusters)
            nprobe = min(index_cfg.ivf_max_probe, nlist // 2)  # Проверяем больше кластеров
            
            quantizer = faiss.IndexFlatL2(index_cfg.dimension)
            index = faiss.IndexIVFFlat(quantizer, 
                                       index_cfg.dimension, 
                                       nlist, 
                                       faiss.METRIC_L2)
            index.nprobe = nprobe
            
        elif index_type == "ivfpq":
            # Экономия памяти, но снижение точности
            nlist = min(int(self._total_documents ** index_cfg.ivfpq_scaling_factor), index_cfg.ivfpq_max_clusters)
            
            quantizer = faiss.IndexFlatL2(index_cfg.dimension)
            index = faiss.IndexIVFPQ(quantizer, 
                                     index_cfg.dimension, 
                                     nlist, 
                                     index_cfg.ivfpq_n_subvectors, 
                                     index_cfg.ivfpq_n_bits)
            index.nprobe = min(index_cfg.ivfpq_n_subvectors, nlist // 2)
            
        if self._device != 'cpu' and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
            
        return index

    def add_documents(self, 
                      embeddings: np.ndarray,
                      documents: List[RagDocument]):
        """
        Adds embeddings, documents, and associated metadata to the store.
        """
        assert len(embeddings) == len(documents), "Length mismatch"
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self._documents.extend(documents)

    def add_raw_documents(self,
                          embeddings: np.ndarray,
                          contents: List[str],
                          metadatas: List[Dict[str, Any]]):
        """
        Adds embeddings, documents, and associated metadata to the store.
        """
        assert len(embeddings) == len(contents) == len(metadatas), "Length mismatch"
        documents = [RagDocument(chunks=content, metadata=metadata)
                     for content, metadata in zip(contents, metadatas)]
        self.add_documents(embeddings, documents)

    def search(self, 
               query_embeddings: np.ndarray, 
               k: int = 5, 
               l2_normalization = True) -> List[List[Tuple[str, Dict]]]:
        """
        Returns top-k matching documents based on the query embedding.
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.astype("float32").reshape(1, -1)

        if l2_normalization:
            faiss.normalize_L2(query_embeddings)

        distances, indices = self.index.search(query_embeddings, k)

        results = []
        for batch_id in range(query_embeddings.shape[0]):
            batch_results = []
            for dist, idx in zip(distances[batch_id], indices[batch_id]):
                if idx < len(self._documents):
                    metadata = self._metadata[idx]
                    metadata['distance'] = float(dist)
                    batch_results.append((self._documents[idx], metadata))
            results.append(batch_results)
        return results

    def save(self, path: str = None):
        """
        Saves FAISS index and metadata to disk.
        """
        db_dir = path if path else self._config.get('rag').get('vector_db').get('dir')
        faiss.write_index(self.index, f"{db_dir}/{self._FAISS_INDEX_FILENAME}")
        with open(f"{db_dir}/{self._METADATA_FILENAME}", "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "metadata": self._metadata
            }, f)

    @classmethod
    def load(cls, path: str = None) -> "VectorStore":
        """
        Loads FAISS index and metadata from disk.
        """
        db_cfg = cls._config.get('rag').get('vector_db')
        db_dir = path if path else db_cfg.get('dir')
        index = faiss.read_index(f"{db_dir}/{cls._FAISS_INDEX_FILENAME}")
        with open(f"{db_dir}/{cls._METADATA_FILENAME}", "rb") as f:
            data = pickle.load(f)

        store = cls(db_cfg.get('embed_dim'))
        store.index = index
        store._documents = data["documents"]
        store._metadata = data["metadata"]
        return store