import logging
import torch

class Config:
    """Configuration for the QA system."""
    data_path = "ServiceBook.xlsx"  # Path to the spreadsheet
    embedding_model = "intfloat/multilingual-e5-large"  # Embedding model
    index_dir = "index_lib"  # Directory for FAISS index and processed data
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Device for computations
    batch_size = 16  # Batch size for embedding generation
    max_length = 512  # Max token length for embeddings
    top_k = 5  # Number of top results to retrieve

def setup_logging(log_level="INFO"):
    """Set up logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
