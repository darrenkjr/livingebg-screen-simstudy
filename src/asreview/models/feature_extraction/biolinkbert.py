__all__ = ["BioLinkBert"]
import torch 
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 
import numpy as np 
from asreview.models.feature_extraction.base import BaseFeatureExtraction

class BioLinkBert(BaseFeatureExtraction):
    """Biolinkbert feature extraction technique.
    
    Biolinkbert is a transformer based model trained on PubMed and citation links"""

    name = "biolinkbert"
    label = "biolinkbert"

    def __init__(
        self,
        *args,
        model_name="michiyasunaga/BioLinkBERT-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        split_ta = 0, 
        batch_size = 32,
        **kwargs
    ):
        super(BioLinkBert, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.split_ta = split_ta
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')


    def transform(self, texts):
        texts = texts.tolist() if isinstance(texts, np.ndarray) else texts
            
        print("Encoding texts using biolinkbert, this may take a while...")
        features = []
        
        # Create progress bar for total texts
        with tqdm(
            total=len(texts), 
            desc="Encoding texts with biolinkbert",
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
                    max_length=512  # Keep consistent with biolinkbert's longer context
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_features = outputs.last_hidden_state[:, 0].cpu().numpy()
                    features.append(batch_features)
                pbar.update(len(batch_texts))

        X = np.vstack(features)
        return X