__all__ = ["modernbert"]



import torch 
from transformers import AutoTokenizer, ModernBertModel
from tqdm import tqdm 
import numpy as np 
from asreview.models.feature_extraction.base import BaseFeatureExtraction

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
        model_name="answerdotai/ModernBERT-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        split_ta = 0, 
        batch_size = 32,
        max_length = 1024, #by default 
        **kwargs
    ):
        super(modernbert, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.split_ta = split_ta
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ModernBertModel.from_pretrained(self.model_name).to(self.device)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')


    def transform(self, texts):
        texts = texts.tolist() if isinstance(texts, np.ndarray) else texts
            
        print("Encoding texts using ModernBERT, this may take a while...")
        features = []
        
        # Create progress bar for total texts
        with tqdm(
            total=len(texts), 
            desc="Encoding texts with ModernBERT",
            position=0,
            leave=True,
            ncols=80
        ) as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=1024  # Keep consistent with ModernBERT's longer context
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_features = outputs.last_hidden_state[:, 0].cpu().numpy()
                    features.append(batch_features)

                pbar.update(len(batch_texts))

        X = np.vstack(features)
        return X