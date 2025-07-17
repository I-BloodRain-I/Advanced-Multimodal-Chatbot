from modules.rag.vector_db.pinecone_db import PineconeDatabase
from shared.config import Config    


def load_vector_db(db_name: str, **kwargs) -> PineconeDatabase:
    """
    Loads and returns a vector database instance based on the specified database name.

    Args:
        db_name (str): The name of the vector database to load (e.g., 'pinecone').
        **kwargs: Additional keyword arguments to override or supplement the database config.

    Returns:
        PineconeDatabase: An initialized instance of the specified vector database.

    Raises:
        ValueError: If the provided db_name is not supported.
    """
    cfg = Config().get('rag.vector_db')
    if db_name == 'pinecone':
        args = cfg.get('pinecone')
        # Allow overriding or adding parameters via kwargs
        args.update(kwargs)
        return PineconeDatabase(**args)
    else:
        raise ValueError(f"Unsupported vector DB type: {db_name}")