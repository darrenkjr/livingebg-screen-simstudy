__all__ = ["modernbert"]

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    MODERNBERT_AVAILABLE = False
else:
    MODERNBERT_AVAILABLE = True

from asreview.models.feature_extraction.base import BaseFeatureExtraction


def _check_mb():
    if not MODERNBERT_AVAILABLE:
        raise ImportError("Install transformers package to use ModernBERT.")


class modernbert(BaseFeatureExtraction):
    """ModernBERT feature extraction technique.
    
    ModernBERT is a BERT model trained on modern scientific papers, making it
    particularly effective for scientific document embeddings and classification.

    Parameters
    ----------
    model_name : str, optional
        The ModernBERT model to use.
        Default: 'allenai/modernbert'
    """

    name = "modernbert"
    label = "ModernBERT"

    def __init__(
        self,
        *args,
        model_name="allenai/modernbert",
        **kwargs
    ):
        super(modernbert, self).__init__(*args, **kwargs)
        self.model_name = model_name

    def transform(self, texts):
        _check_mb()

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        # Preprocess texts
        text_batch = [text for text in texts]  # Assuming texts are already title + abstract
        inputs = tokenizer(
            text_batch, 
            padding=True, 
            truncation=True,
            return_tensors="pt", 
            return_token_type_ids=True,  # ModernBERT uses token_type_ids
            max_length=512
        )

        print("Encoding texts using ModernBERT, this may take a while...")
        # Get embeddings from the model
        output = model(**inputs)
        # Take the first token ([CLS]) embedding as the document embedding
        X = output.last_hidden_state[:, 0, :].detach().numpy()

        return X