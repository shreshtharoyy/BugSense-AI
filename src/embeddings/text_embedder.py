from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, text:str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        return embedding