import logging

import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class Retriever:
    def __init__(self, config, data_frame):
        self.config = config
        self.data = data_frame
        if 'searchable_text' not in self.data.columns:
            raise ValueError("The data must contain a 'searchable_text' column.")
        self.device = config.device
        self.index = faiss.read_index(f"{config.index_dir}/faiss_index.index")
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        self.model = AutoModel.from_pretrained(config.embedding_model)
        self.model.to(self.device)
        self.model.eval()

    def _create_embeddings(self, texts):
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Query Embeddings"):
                batch_texts = texts[i:i + self.config.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                embeddings.append(last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)

    def get_relevant_docs(self, query, k=5):
        query_embedding = self._create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.data):
                match_percentage = round((1 - dist) * 100, 2)
                results.append({
                    "text": self.data.iloc[idx]['searchable_text'],
                    "match_percentage": match_percentage
                })
        return results


class QASystem:
    def __init__(self, config, data_frame):
        self.config = config
        self.retriever = Retriever(config, data_frame)

    def answer_question(self, question):
        try:
            results = self.retriever.get_relevant_docs(question, k=self.config.top_k)
            if not results:
                return "לא נמצאה תוצאה מתאימה לשאלה שלך."

            # Split fields using | and format output
            formatted_results = "\n\n".join([
                (
                    f"רופא: {fields[0]} {fields[1]}\n"
                    f"תואר: {fields[2]}\n"
                    f"מספר רשיון: {fields[3]}\n"
                    f"תחום התמחות: {fields[4]}\n"
                    f"תת-התמחות: {fields[5] or 'מידע חסר'}\n"
                    f"עיר: {fields[6]}\n"
                    f"כתובת: {fields[7]}\n"
                    f"טלפונים: {fields[8]}\n"
                    #f"% התאמה: {round(result['match_percentage'], 2)}%"
                )
                for result in results
                for fields in [result['text'].split('|')]
            ])
            return f"בבקשה, הנה מידע רלוונטי לשאלה שלך:\n\n{formatted_results}"
        except Exception as e:
            logging.error(f"Error in answer generation: {e}")
            return "שגיאה בתהליך השאלה. נסה שנית."
