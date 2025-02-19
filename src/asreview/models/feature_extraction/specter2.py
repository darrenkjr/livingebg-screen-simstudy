__all__ = ["specter2"]

try:
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    import numpy as np 
    import torch
except ImportError:
    SPEC2_AVAILABLE = False
else:
    SPEC2_AVAILABLE = True

from asreview.models.feature_extraction.base import BaseFeatureExtraction


def _check_st():
    if not SPEC2_AVAILABLE:
        raise ImportError("Install sentence-transformers package to use Sentence BERT.")


class SPECTER2(BaseFeatureExtraction):
    """SPECTER2 feature extraction technique with classification adapter."""
    
    name = "specter2"
    label = "SPECTER2"

    def __init__(
        self,
        *args,
        base_model="allenai/specter2_base",
        adapter_model="allenai/specter2_classification",
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super(SPECTER2, self).__init__(split_ta=0, use_keywords=0)
        self.base_model = base_model
        self.adapter_model = adapter_model
        self.batch_size = batch_size
        self.device = device
        
        _check_st()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoAdapterModel.from_pretrained(self.base_model).to(self.device)
        
        # Load and activate the classification adapter
        self.model.load_adapter(
            self.adapter_model, 
            source="hf", 
            load_as="classification", 
            set_active=True
        )
        self.model.eval()

    def transform(self, texts):
        """Transform texts into feature vectors."""
        if not isinstance(texts, np.ndarray):
            texts = np.array(texts)

        features = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        print(f"Encoding {len(texts)} texts using SPECTER2...")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Concatenate title and abstract with sep_token
            text_batch = [
                title.strip() + self.tokenizer.sep_token + abstract.strip()
                for text in batch_texts
                for title, abstract in [str(text).split('\n', 1) + [''] 
                    if len(str(text).split('\n', 1)) == 1 
                    else str(text).split('\n', 1)]
            ]

            # Preprocess the input
            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.last_hidden_state[:, 0].cpu().numpy()
                features.append(batch_features)

            if (i // self.batch_size) % 100 == 0:
                print(f"Processed {i//self.batch_size}/{total_batches} batches")

        # Combine all batches
        X = np.vstack(features)
        return X