from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, text:Union[str, List[str]], batch_size: int=16) -> np.ndarray:
        embedding = self.model.encode(text, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

        return embedding