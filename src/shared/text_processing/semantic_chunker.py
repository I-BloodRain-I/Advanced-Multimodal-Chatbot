"""
Semantic text chunking based on sentence similarity.

This module provides the SemanticChunker class that intelligently splits text
into chunks based on semantic similarity between sentences. It uses sentence
embeddings to determine when to break chunks, ensuring related content stays
together while respecting token limits.

The chunker uses a greedy grouping algorithm with cosine similarity to maintain
semantic coherence within chunks, making it ideal for RAG systems and document
processing where context preservation is important.
"""

from typing import List
import logging

import numpy as np
import torch
import torch.nn.functional as F
import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from shared.utils import get_torch_device

logger = logging.getLogger(__name__)
# Download once
nltk.download("punkt_tab", quiet=True)


class SemanticChunker:
    """
    A similarity-aware sentence chunker that greedily groups semantically related sentences
    into token-length-constrained chunks.

    Args:
        model_name: Name of the SentenceTransformer model to load.
        chunk_token_limit: Maximum number of tokens allowed per chunk.
        similarity_threshold: Minimum cosine similarity to keep sentences in the same chunk.
        min_chunk_tokens: Minimum number of tokens required for a chunk; small chunks may be merged.
        batch_size: Batch size for embedding sentences.
        device_name: Name of the torch device (e.g., 'cuda' or 'cpu').
        show_progress: Whether to display progress bars during processing.
    """

    def __init__(self,
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 chunk_token_limit: int = 380,
                 similarity_threshold: float = 0.75,
                 min_chunk_tokens: int = 120,
                 batch_size: int = 64,
                 device_name: str = 'cuda',
                 show_progress: bool = False):
        logger.info("Initializing SemanticChunker...")

        # Load configuration values
        self.chunk_limit = chunk_token_limit
        self.sim_threshold = similarity_threshold
        self.min_tokens = min_chunk_tokens
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.device = get_torch_device(device_name)

        self._validate_parameters()

        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)
        logger.info("SemanticChunker initialized successfully.")

    def _validate_parameters(self):
        """Validate initialization parameters for logical consistency and value ranges."""
        if self.chunk_limit <= 0:
            raise ValueError(f"chunk_token_limit must be positive, got {self.chunk_limit}")
        if not -1 <= self.sim_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between -1 and 1, got {self.sim_threshold}")
        if self.min_tokens < 0:
            raise ValueError(f"min_chunk_tokens must be non-negative, got {self.min_tokens}")
        if self.min_tokens >= self.chunk_limit:
            raise ValueError(f"min_chunk_tokens ({self.min_tokens}) must be less than chunk_token_limit ({self.chunk_limit})")

    def _load_model(self, model_name: str) -> SentenceTransformer:
        try:
            logger.info(f"Loading model '{model_name}' on device '{self.device}'...")
            model = SentenceTransformer(model_name, device=self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _load_tokenizer(self, tokenizer_name: str) -> AutoTokenizer:
        try:
            logger.info(f"Loading tokenizer '{tokenizer_name}'...")
            return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            raise

    def split(self, text: str) -> List[str]:
        """
        Split input text into semantically coherent chunks.

        Args:
            text: The raw input text to be chunked.

        Returns:
            A list of chunked strings, each fitting the semantic
            and token constraints.
        """
        if not text.strip():
            logger.warning("Empty or whitespace-only input received.")
            return []
        
        sentences = self._split_sentences(text)
        if not sentences:
            logger.warning("No valid sentences extracted from text.")
            return []
        
        sent_embeddings = self._embed(sentences)
        raw_chunks = self._greedy_group(sentences, sent_embeddings)
        return self._merge_small(raw_chunks)

    def _merge_small(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are shorter than `min_chunk_tokens`."""
        if not chunks:
            return chunks
        
        merged = [chunks[0]]
        
        for i, chunk in enumerate(chunks[1:], 1):
            chunk_tokens = self._count_tokens(chunk)
            prev = merged[-1]
            prev_tokens = self._count_tokens(prev)
            
            # Try to merge if at least one of the chunks is smaller than the minimum
            # and the total size does not exceed the limit
            if (prev_tokens < self.min_tokens or chunk_tokens < self.min_tokens) and \
            (prev_tokens + chunk_tokens <= self.chunk_limit):
                
                merged.pop()
                merged_chunk = f"{prev} {chunk}"
                merged.append(merged_chunk)
                logger.debug(f"Merging chunk {i} ({chunk_tokens} tokens) with previous ({prev_tokens} tokens)")
            else:
                merged.append(chunk)
                logger.debug(f"Keeping chunk {i} separate ({chunk_tokens} tokens)")
        
        return merged

    def _split_sentences(self, text: str) -> List[str]:
        """Use NLTK to split text into sentences."""
        return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]

    def _embed(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress) 

    def _greedy_group(self, sentences: List[str], embeddings: np.ndarray) -> List[str]:
        """Group sentences into semantically coherent chunks using a greedy strategy."""
        chunks, current_chunk, current_embeddings = [], [], []
        current_tokens = 0

        iterator = zip(sentences, embeddings)
        if self.show_progress:
            iterator =  tqdm(iterator, total=len(sentences), desc="Chunking", unit="sentence")

        for i, (sentence, embedding) in enumerate(iterator):
            sentence_tokens = self._count_tokens(sentence)

            # 1) too long to ever fit -> emit as standalone
            if sentence_tokens >= self.chunk_limit:
                logger.debug(f"Sentence {i} too long ({sentence_tokens} tokens). Emitting standalone.")
                self._append_chunk(chunks, current_chunk)
                chunks.append(sentence)
                current_chunk, current_embeddings, current_tokens = [], [], 0
                continue

            # 2) empty chunk -> start
            if not current_chunk:
                current_chunk, current_embeddings, current_tokens = [sentence], [embedding], sentence_tokens
                continue

            # 3) Compute cosine similarity between current sentence and 
            #    the centroid of current chunk embeddings.
            centroid = torch.mean(torch.tensor(np.array(current_embeddings)), dim=0, keepdim=True)
            sim = F.cosine_similarity(centroid, torch.from_numpy(embedding), dim=1).item()

            # 4) decide where to place the sentence
            fits_length = current_tokens + sentence_tokens <= self.chunk_limit
            fits_semantics = sim >= self.sim_threshold
            if fits_length and fits_semantics:
                # append to current chunk
                current_chunk.append(sentence)
                current_embeddings.append(embedding)
                current_tokens += sentence_tokens
            else:
                # start a new chunk
                logger.debug(f"Breaking chunk at sentence {i} (sim={sim:.3f}, tokens={current_tokens + sentence_tokens})")
                self._append_chunk(chunks, current_chunk)
                current_chunk, current_embeddings, current_tokens = [sentence], [embedding], sentence_tokens

        self._append_chunk(chunks, current_chunk)
        return chunks

    def _append_chunk(self, container: list, chunk: list):
        if chunk:
            container.append(" ".join(chunk))

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))