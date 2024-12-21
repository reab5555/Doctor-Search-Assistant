import logging
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from src.config import Config, setup_logging
from tqdm import tqdm
from pathlib import Path

def prepare_searchable_text(data_frame):
    """Create a `searchable_text` column in Hebrew for data enrichment."""
    def format_address(row):
        # Combine רחוב and מספר בית into כתובת
        street = row.get('רחוב', '')
        house_number = row.get('מספר בית', '')
        return f"{street} {house_number}".strip()

    def format_phones(phones):
        # Format phone numbers, handling lists or strings
        if isinstance(phones, list):
            return ", ".join(phones)
        if isinstance(phones, str):
            return phones.strip()
        return "מידע חסר"

    data_frame['כתובת'] = data_frame.apply(format_address, axis=1)
    data_frame['טלפונים'] = data_frame['טלפונים'].apply(format_phones)

    # Use | as the delimiter for searchable_text
    data_frame['searchable_text'] = data_frame.apply(
        lambda row: (
            f"{row.get('שם פרטי', '')}|"
            f"{row.get('שם משפחה', '')}|"
            f"{row.get('תואר', '')}|"
            f"{row.get('מספר רשיון', '')}|"
            f"{row.get('התמחות', '')}|"
            f"{row.get('תת-התמחות', '')}|"
            f"{row.get('עיר', '')}|"
            f"{row.get('כתובת', '')}|"
            f"{row.get('טלפונים', '')}"
        ).strip(),
        axis=1
    )
    return data_frame

def generate_embeddings(model, tokenizer, texts, device, batch_size):
    """Generate embeddings for a list of texts."""
    embeddings = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def main():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    config = Config()

    # Load data
    logger.info("Loading data from Excel file...")
    data_frame = pd.read_excel(config.data_path)

    # Enrich data with searchable_text
    logger.info("Creating searchable_text column...")
    data_frame = prepare_searchable_text(data_frame)
    logger.info(f"Sample searchable_text: {data_frame['searchable_text'].head(1).tolist()}")

    # Generate embeddings
    logger.info("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
    model = AutoModel.from_pretrained(config.embedding_model, output_hidden_states=True)

    logger.info("Generating document embeddings...")
    document_texts = data_frame['searchable_text'].tolist()
    document_embeddings = generate_embeddings(
        model, tokenizer, document_texts, config.device, config.batch_size
    )

    # Create FAISS index
    logger.info("Creating FAISS index...")
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(document_embeddings)
    index.add(document_embeddings)

    # Save FAISS index
    index_dir = Path(config.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_index_path = index_dir / "faiss_index.index"
    faiss.write_index(index, str(faiss_index_path))
    logger.info(f"FAISS index saved to {faiss_index_path}.")

    # Save processed data
    processed_data_path = index_dir / "processed_data.json"
    data_frame.to_json(processed_data_path, orient='records', lines=True, force_ascii=False)
    logger.info(f"Processed data saved to {processed_data_path}.")

if __name__ == "__main__":
    main()
