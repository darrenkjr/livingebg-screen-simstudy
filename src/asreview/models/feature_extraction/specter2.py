__all__ = ["specter2"]


from transformers import AutoTokenizer
import torch
from adapters import AutoAdapterModel
import numpy as np 
from tqdm import tqdm
from asreview.models.feature_extraction.base import BaseFeatureExtraction

class specter2(BaseFeatureExtraction):
    """specter2 feature extraction technique with classification adapter."""
    
    name = "specter2"
    label = "specter2"

    def __init__(
        self,
        *args,
        base_model="allenai/specter2_base",
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super(specter2, self).__init__(split_ta=1, use_keywords=0)
        self.base_model = base_model
        self.batch_size = batch_size
        self.device = device
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoAdapterModel.from_pretrained(self.base_model).to(self.device)

    def fit_transform(self, texts, titles=None, abstracts=None, keywords=None):
        if self.split_ta > 0:
            if titles is None or abstracts is None:
                raise ValueError(
                    "Error: if splitting titles and abstracts, supply them!"
                )
                
            print('Preprocessing texts...')
            # Convert numpy arrays to lists for the tokenizer
            titles = titles.tolist() if isinstance(titles, np.ndarray) else titles
            abstracts = abstracts.tolist() if isinstance(abstracts, np.ndarray) else abstracts
            
            # Combine with sep token
            sep_token = self.tokenizer.sep_token
            texts = [f"{title}{sep_token}{abstract}" for title, abstract in zip(titles, abstracts)]
            print(f'Combined {len(texts)} texts with separator token')
        else:
            # If texts is numpy array, convert to list
            texts = texts.tolist() if isinstance(texts, np.ndarray) else texts

        X = self.transform(texts)
        return X
            

    def transform(self, texts):
        """Transform texts into feature vectors."""
        features = []
        print('Encoding texts with SPECTER2')
        # Create progress bar for total texts
        with tqdm(
            total=len(texts), 
            desc="Encoding texts with SPECTER2",
            position=0,  # Keep at position 0
            leave=True,  # Leave the progress bar
            ncols=80  # Fixed width
        ) as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_features = outputs.last_hidden_state[:, 0].cpu().numpy()
                    features.append(batch_features)

                # Update progress for each text in batch
                pbar.update(len(batch_texts))

        X = np.vstack(features)
        return X