"""Module 4: ClinicalBERT Embedding Generation

This module generates text embeddings for discharge summaries
using ClinicalBERT transformer model.
"""

import pandas as pd
import numpy as np
import logging
import torch
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available, will use PCA for dimensionality reduction")

logger = logging.getLogger(__name__)


class ClinicalBERTEmbedder:
    """Generate ClinicalBERT embeddings for clinical text."""
    
    def __init__(self, config: Dict):
        """Initialize ClinicalBERT embedder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.embed_config = config['embeddings']
        
        self.model_name = self.embed_config['model_name']
        self.max_length = self.embed_config['max_length']
        self.batch_size = self.embed_config['batch_size']
        self.device = self.embed_config['device']
        self.pooling_strategy = self.embed_config['pooling_strategy']
        
        # Check device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = 'cpu'
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self) -> None:
        """Load ClinicalBERT model and tokenizer."""
        logger.info(f"Loading ClinicalBERT model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading ClinicalBERT model: {e}")
            logger.info("Please ensure you have internet connection for first-time download")
            raise
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within max token length.
        
        Args:
            text: Input text
            
        Returns:
            Truncated text
        """
        # Tokenize and truncate
        tokens = self.tokenizer.encode(text, add_special_tokens=True, 
                                      max_length=self.max_length, 
                                      truncation=True)
        
        # Decode back to text
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return truncated_text
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text to embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        elif self.pooling_strategy == 'mean':
            # Mean pooling over all tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
        elif self.pooling_strategy == 'max':
            # Max pooling over all tokens
            embedding = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embeddings.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     desc="Generating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            
            for text in batch_texts:
                embedding = self.encode_text(text)
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensionality.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Reduced embeddings
        """
        dim_config = self.embed_config['dimensionality_reduction']
        method = dim_config['method']
        n_components = dim_config['n_components']
        
        if method == 'none':
            logger.info("Skipping dimensionality reduction")
            return embeddings
        
        logger.info(f"Reducing dimensionality using {method} to {n_components} components")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            logger.info(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.3f}")
        
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP not available, falling back to PCA")
                reducer = PCA(n_components=n_components, random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.1
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reduced_embeddings
    
    def process(self, psych_df: pd.DataFrame) -> tuple:
        """Generate embeddings for all discharge summaries.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            Tuple of (DataFrame with embeddings, embedding matrix)
        """
        logger.info("Starting ClinicalBERT embedding generation")
        
        # Load model
        self.load_model()
        
        # Get discharge summaries
        # texts = psych_df['discharge_summary'].tolist()
        # logger.info(f"Generating embeddings for {len(texts)} discharge summaries")
        # Get discharge summaries column
        if 'discharge_summary' not in psych_df.columns:
            raise KeyError("Column 'discharge_summary' not found in DataFrame")

        texts = psych_df['discharge_summary'].tolist()

        # Convert non-string values to empty string
        clean_texts = []
        for t in texts:
            if isinstance(t, str):
                clean_texts.append(t)
            elif pd.isna(t):
                clean_texts.append("")
            else:
                clean_texts.append(str(t))  # convert float/dict/etc. safely

        logger.info(f"Generating embeddings for {len(clean_texts)} discharge summaries")

        # Use sanitized texts
        texts = clean_texts

        # Generate embeddings
        embeddings = self.encode_batch(texts)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Reduce dimensionality if configured
        reduced_embeddings = self.reduce_dimensionality(embeddings)
        logger.info(f"Final embedding shape: {reduced_embeddings.shape}")
        
        # Add embeddings to dataframe
        for i in range(reduced_embeddings.shape[1]):
            psych_df[f'embedding_{i}'] = reduced_embeddings[:, i]
        
        # Save embeddings
        embeddings_path = self.config['data']['embeddings_path']
        import sys
        sys.path.append('/app')
        from src.utils import save_pickle
        
        save_pickle({
            'embeddings': reduced_embeddings,
            'subject_ids': psych_df['SUBJECT_ID'].tolist(),
            'hadm_ids': psych_df['HADM_ID'].tolist()
        }, embeddings_path)
        
        logger.info(f"Embeddings saved to {embeddings_path}")
        
        return psych_df, reduced_embeddings
