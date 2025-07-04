from typing import List, Tuple, Callable, Union
from src.vectorstore.vector_store import VectorStore
import faiss
import torch

def semantic_search(
    store: VectorStore,
    query_text: Union[str, List[str]],
    embed_fn: Callable[[List[str]], torch.Tensor],  # Function: List[str] -> torch.Tensor
    l2_normalization: bool = True,
    top_k: int = 5
) -> List[Tuple[str, dict]]:
    """
    Performs semantic search over a vector store using embedded query text.

    Args:
        store (VectorStore): The vector store to search against.
        query_text (Union[str, List[str]]): The query string or list of query strings.
        embed_fn (Callable[[List[str]], torch.Tensor]): A function that maps a list of strings 
            to a tensor of embeddings.
        l2_normalization (bool, optional): Whether to apply L2 normalization to the query embeddings. 
            Defaults to True.
        top_k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        List[Tuple[str, dict]]: A list of tuples containing matched documents and their associated metadata.
    """
    if isinstance(query_text, str):
        query_text = [query_text]

    embedding = embed_fn(query_text).detach().cpu().numpy()  # Assumes list input, returns np.ndarray

    if l2_normalization:
        faiss.normalize_L2(embedding)
        
    return store.search(embedding, k=top_k)