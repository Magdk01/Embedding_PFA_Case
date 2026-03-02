import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from loguru import logger

MODEL_NAME = "intfloat/multilingual-e5-large"


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Apply average pooling over token embeddings, masked by attention."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel:
    """Wrapper around multilingual-e5-large for text embedding.

    The model expects each input to be prefixed with either "query: " or "passage: "
    depending on the use case. This class handles the prefixing automatically.

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        logger.info("Model loaded successfully")

    def embed(self, texts: list[str], prefix: str = "query") -> list[list[float]]:
        """Generate normalized embeddings for a list of texts.

        Args:
            texts: Raw input texts (without prefix).
            prefix: "query" for questions/symmetric tasks, "passage" for documents/retrieval.

        Returns:
            List of normalized 1024-dimensional embedding vectors.
        """
        prefixed = [f"{prefix}: {t}" for t in texts]
        logger.debug(f"Embedding {len(texts)} texts with prefix='{prefix}'")

        batch_dict = self.tokenizer(prefixed, max_length=512, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

if __name__ == "__main__":
    texts = ["how much protein should a female eat", "hvad er vejret i dag"]
    model = EmbeddingModel()
    embeddings = model.embed(texts,prefix=["passage","query"])
    print(embeddings)