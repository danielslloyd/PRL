"""
ClinVec Integration for ExMed-BERT
Integrates pre-trained ClinVec embeddings with ExMed-BERT model
"""

import logging
import pandas as pd
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List, Set
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class ClinVecLoader:
    """Load and integrate ClinVec embeddings with ExMed-BERT"""
    
    def __init__(self, clinvec_dir: str):
        """
        Initialize ClinVec loader
        
        Args:
            clinvec_dir: Path to directory containing ClinVec files
        """
        self.clinvec_dir = Path(clinvec_dir)
        self.node_mapping = None
        self.embeddings_cache = {}
        
    def load_node_mapping(self) -> pd.DataFrame:
        """Load node mapping from ClinGraph_nodes.csv"""
        if self.node_mapping is None:
            nodes_path = self.clinvec_dir / "ClinGraph_nodes.csv"
            self.node_mapping = pd.read_csv(nodes_path, sep='\t')
            logger.info(f"Loaded {len(self.node_mapping)} node mappings")
        return self.node_mapping
    
    def load_embeddings_by_vocab(self, vocab_type: str) -> Dict[str, torch.Tensor]:
        """
        Load embeddings for a specific vocabulary type
        
        Args:
            vocab_type: Type of vocabulary (icd9cm, icd10cm, phecode, atc, rxnorm, etc.)
            
        Returns:
            Dictionary mapping original codes to embedding tensors
        """
        vocab_key = vocab_type.lower()
        
        if vocab_key in self.embeddings_cache:
            return self.embeddings_cache[vocab_key]
        
        # Load embeddings file
        emb_file = self.clinvec_dir / f"ClinVec_{vocab_key}.csv"
        if not emb_file.exists():
            logger.warning(f"Embeddings file not found: {emb_file}")
            return {}
        
        emb_df = pd.read_csv(emb_file, index_col=0)
        logger.info(f"Loaded {len(emb_df)} {vocab_type} embeddings, dim={emb_df.shape[1]}")
        
        # Load node mapping
        nodes_df = self.load_node_mapping()
        
        # Merge embeddings with node info
        emb_df['node_index'] = emb_df.index
        merged_df = emb_df.merge(nodes_df, on='node_index', how='inner')
        
        # Filter by vocabulary type
        vocab_filtered = merged_df[merged_df['ntype'].str.upper() == vocab_type.upper()]
        
        # Extract original codes and convert to tensors
        embeddings_dict = {}
        for _, row in vocab_filtered.iterrows():
            # Extract original code from node_id (format: "code:vocab")
            original_code = row['node_id'].split(':')[0]
            
            # Get embedding values (exclude metadata columns)
            emb_values = row.drop(['node_index', 'node_id', 'node_name', 'ntype']).values

            # Convert object array to float32 numpy array first
            if emb_values.dtype == object:
                # Convert each element to float, then create float32 array
                emb_values = np.array([float(val) for val in emb_values], dtype=np.float32)
            else:
                emb_values = emb_values.astype(np.float32)

            embeddings_dict[original_code] = torch.tensor(emb_values, dtype=torch.float32)
        
        self.embeddings_cache[vocab_key] = embeddings_dict
        logger.info(f"Created embeddings dict with {len(embeddings_dict)} {vocab_type} codes")
        
        return embeddings_dict
    
    def resize_embeddings(self, embeddings: Dict[str, torch.Tensor], target_dim: int,
                         method: str = "auto") -> Dict[str, torch.Tensor]:
        """
        Intelligently resize embeddings to match model dimensions

        Args:
            embeddings: Dictionary of embeddings
            target_dim: Target embedding dimension
            method: Resizing method - "auto", "truncate", "pca", "learned_projection", "pad_smart", "pad_random"

        Returns:
            Resized embeddings dictionary
        """
        if not embeddings:
            return embeddings

        sample_emb = next(iter(embeddings.values()))
        current_dim = sample_emb.shape[0]

        if current_dim == target_dim:
            return embeddings

        # Automatically select method if not specified
        if method == "auto":
            method = self._select_resize_method(current_dim, target_dim)

        logger.info(f"Resizing embeddings from {current_dim} to {target_dim} dimensions using method: {method}")

        # Apply the selected method
        if method == "truncate":
            return self._resize_truncate(embeddings, target_dim)
        elif method == "pca":
            return self._resize_pca(embeddings, current_dim, target_dim)
        elif method == "learned_projection":
            return self._resize_learned_projection(embeddings, current_dim, target_dim)
        elif method == "pad_smart":
            return self._resize_pad_smart(embeddings, current_dim, target_dim)
        elif method == "pad_random":
            return self._resize_pad_random(embeddings, current_dim, target_dim)
        else:
            logger.warning(f"Unknown resize method: {method}. Falling back to simple resize.")
            return self._resize_simple(embeddings, current_dim, target_dim)

    def _select_resize_method(self, current_dim: int, target_dim: int) -> str:
        """
        Automatically select the best resizing method based on dimension ratio

        Args:
            current_dim: Current embedding dimension
            target_dim: Target embedding dimension

        Returns:
            Selected method name
        """
        ratio = target_dim / current_dim

        if ratio == 1.0:
            return "none"
        elif ratio < 1.0:  # Compression needed
            compression_ratio = ratio
            if compression_ratio > 0.8:
                return "truncate"  # Preserve ClinVec structure
            else:
                return "pca"  # Maximize information retention
        else:  # Expansion needed
            expansion_ratio = ratio
            if expansion_ratio < 1.2:
                return "learned_projection"  # Structured expansion
            else:
                return "pad_smart"  # Let transformer adapt

    def _resize_truncate(self, embeddings: Dict[str, torch.Tensor], target_dim: int) -> Dict[str, torch.Tensor]:
        """Simple truncation - preserve first dimensions"""
        logger.info("Using truncation method - preserving ClinVec structure")
        return {code: emb[:target_dim] for code, emb in embeddings.items()}

    def _resize_pca(self, embeddings: Dict[str, torch.Tensor], current_dim: int, target_dim: int) -> Dict[str, torch.Tensor]:
        """PCA-based compression - maximize information retention"""
        logger.info("Using PCA method - maximizing information retention")

        # Stack all embeddings for PCA
        embedding_list = list(embeddings.values())
        stacked_embeddings = torch.stack(embedding_list)  # Shape: [vocab_size, current_dim]

        # Compute PCA using SVD
        U, S, V = torch.pca_lowrank(stacked_embeddings, q=target_dim)

        # V contains the principal components - shape: [current_dim, target_dim]
        projection_matrix = V[:, :target_dim]

        # Project each embedding
        resized = {}
        for code, emb in embeddings.items():
            resized[code] = emb @ projection_matrix

        logger.info(f"PCA explained variance ratio: {(S[:target_dim].sum() / S.sum()).item():.3f}")
        return resized

    def _resize_learned_projection(self, embeddings: Dict[str, torch.Tensor], current_dim: int, target_dim: int) -> Dict[str, torch.Tensor]:
        """Learned projection - structured expansion"""
        logger.info("Using learned projection method - structured expansion")

        # Create a projection layer
        projection = torch.nn.Linear(current_dim, target_dim, bias=False)

        # Smart initialization to preserve ClinVec structure
        with torch.no_grad():
            if target_dim >= current_dim:
                # Expanding: preserve original dimensions in first part
                projection.weight[:current_dim, :current_dim] = torch.eye(current_dim)
                # Initialize additional dimensions with small random weights
                if target_dim > current_dim:
                    torch.nn.init.normal_(projection.weight[current_dim:, :], mean=0, std=0.01)
                    torch.nn.init.normal_(projection.weight[:current_dim, current_dim:], mean=0, std=0.01)
            else:
                # This shouldn't happen with learned_projection, but handle gracefully
                torch.nn.init.orthogonal_(projection.weight)

        # Apply projection
        resized = {}
        with torch.no_grad():
            for code, emb in embeddings.items():
                resized[code] = projection(emb)

        return resized

    def _resize_pad_smart(self, embeddings: Dict[str, torch.Tensor], current_dim: int, target_dim: int) -> Dict[str, torch.Tensor]:
        """Smart padding - structured expansion for large dimension increases"""
        logger.info("Using smart padding method - letting transformer adapt")

        padding_size = target_dim - current_dim

        resized = {}
        for code, emb in embeddings.items():
            # Calculate padding based on embedding statistics
            emb_std = emb.std().item()
            emb_mean = emb.mean().item()

            # Create structured padding
            if padding_size <= current_dim:
                # Small expansion: echo part of the original embedding with noise
                echo_size = padding_size
                echo_indices = torch.randperm(current_dim)[:echo_size]
                padding = emb[echo_indices] + torch.normal(0, emb_std * 0.1, size=(echo_size,))
            else:
                # Large expansion: echo full embedding + random
                echo_emb = emb + torch.normal(0, emb_std * 0.1, size=(current_dim,))
                remaining_size = padding_size - current_dim
                random_padding = torch.normal(emb_mean, emb_std * 0.1, size=(remaining_size,))
                padding = torch.cat([echo_emb, random_padding])

            resized[code] = torch.cat([emb, padding])

        return resized

    def _resize_pad_random(self, embeddings: Dict[str, torch.Tensor], current_dim: int, target_dim: int) -> Dict[str, torch.Tensor]:
        """Random padding - simple expansion"""
        logger.info("Using random padding method")

        padding_size = target_dim - current_dim

        resized = {}
        for code, emb in embeddings.items():
            padding = torch.normal(0, 0.01, size=(padding_size,))
            resized[code] = torch.cat([emb, padding])

        return resized

    def _resize_simple(self, embeddings: Dict[str, torch.Tensor], current_dim: int, target_dim: int) -> Dict[str, torch.Tensor]:
        """Fallback to original simple method"""
        resized = {}
        for code, emb in embeddings.items():
            if current_dim > target_dim:
                resized[code] = emb[:target_dim]
            else:
                padding = torch.normal(0, 0.01, size=(target_dim - current_dim,))
                resized[code] = torch.cat([emb, padding])
        return resized


class HierarchicalCodeMatcher:
    """Handle hierarchical matching for medical codes (ICD-9, ICD-10, ATC)"""

    def __init__(self):
        self.icd10_pattern = re.compile(r'^[A-Z]\d{2}\.?\d*$')
        self.icd9_pattern = re.compile(r'^\d{3}\.?\d*$')
        # ATC codes: Letter + 2 digits + at least 1 more letter for ATC3+
        # ATC5: 7 chars like A01AA01 (Level 1-5)
        # ATC4: 5 chars like A01AA (Level 1-4)
        # ATC3: 4 chars like A01A (Level 1-3)
        # Must have at least one letter after the first 3 chars to distinguish from ICD
        self.atc_pattern = re.compile(r'^[A-Z]\d{2}[A-Z][A-Z]?\d{0,2}$')
    
    def is_icd10_code(self, code: str) -> bool:
        """Check if code follows ICD-10 format"""
        return bool(self.icd10_pattern.match(code))

    def is_icd9_code(self, code: str) -> bool:
        """Check if code follows ICD-9 format"""
        return bool(self.icd9_pattern.match(code))

    def is_atc_code(self, code: str) -> bool:
        """Check if code follows ATC format"""
        return bool(self.atc_pattern.match(code))
    
    def get_code_hierarchy(self, code: str) -> List[str]:
        """
        Get hierarchical variants of a code, from most specific to most general

        Examples:
            ICD-10: E11.65 -> ['E11.65', 'E11.6', 'E11']
            ATC5: A01AA01 -> ['A01AA01', 'A01AA', 'A01A', 'A01']
        """
        hierarchy = [code]

        # Check ICD codes first (more specific due to decimal point)
        if self.is_icd10_code(code):
            hierarchy.extend(self._get_icd10_hierarchy(code))
        elif self.is_icd9_code(code):
            hierarchy.extend(self._get_icd9_hierarchy(code))
        elif self.is_atc_code(code):
            hierarchy.extend(self._get_atc_hierarchy(code))

        return hierarchy
    
    def _get_icd10_hierarchy(self, code: str) -> List[str]:
        """Generate ICD-10 hierarchy: E11.65 -> E11.6 -> E11"""
        hierarchy = []
        
        # Remove decimal if present
        clean_code = code.replace('.', '')
        
        # E11.65 -> E1165, E116, E11
        for i in range(len(clean_code) - 1, 2, -1):  # Stop at 3 chars (E11)
            parent = clean_code[:i]
            
            # Add decimal back for 4+ character codes
            if len(parent) > 3:
                parent = parent[:3] + '.' + parent[3:]
            
            if parent != code and parent not in hierarchy:
                hierarchy.append(parent)
        
        return hierarchy
    
    def _get_icd9_hierarchy(self, code: str) -> List[str]:
        """Generate ICD-9 hierarchy: 250.01 -> 250.0 -> 250"""
        hierarchy = []
        
        # Remove decimal if present  
        clean_code = code.replace('.', '')
        
        # 25001 -> 2500, 250
        for i in range(len(clean_code) - 1, 2, -1):  # Stop at 3 chars
            parent = clean_code[:i]
            
            # Add decimal back for 4+ character codes
            if len(parent) > 3:
                parent = parent[:3] + '.' + parent[3:]
            
            if parent != code and parent not in hierarchy:
                hierarchy.append(parent)
        
        return hierarchy

    def _get_atc_hierarchy(self, code: str) -> List[str]:
        """
        Generate ATC hierarchy from specific to general

        ATC hierarchy levels:
        - Level 5 (7 chars): A01AA01 - Chemical substance
        - Level 4 (5 chars): A01AA - Chemical subgroup
        - Level 3 (4 chars): A01A - Pharmacological subgroup
        - Level 2 (3 chars): A01 - Therapeutic subgroup

        Example: A01AA01 -> A01AA -> A01A -> A01
        """
        hierarchy = []

        # ATC codes don't have decimals
        if len(code) == 7:  # ATC5: A01AA01
            hierarchy.append(code[:5])  # ATC4: A01AA
            hierarchy.append(code[:4])  # ATC3: A01A
            hierarchy.append(code[:3])  # ATC2: A01
        elif len(code) == 5:  # ATC4: A01AA
            hierarchy.append(code[:4])  # ATC3: A01A
            hierarchy.append(code[:3])  # ATC2: A01
        elif len(code) == 4:  # ATC3: A01A
            hierarchy.append(code[:3])  # ATC2: A01
        # If len(code) == 3 (ATC2), no parents (already most general in our system)

        return hierarchy

    def find_cousin_average_embedding(
        self,
        novel_code: str,
        available_embeddings: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Find cousin codes (siblings with same parent) and return their average embedding

        For ATC codes, cousins are codes that share the same parent but differ in the final digits.
        Example: For A01AA01, cousins are A01AA02, A01AA03, etc. (all share parent A01AA)

        Args:
            novel_code: The code without an embedding
            available_embeddings: Dictionary of available embeddings

        Returns:
            Average embedding of cousins, or None if no cousins found
        """
        if not self.is_atc_code(novel_code):
            return None

        # Get the parent code (one level up)
        hierarchy = self.get_code_hierarchy(novel_code)
        if len(hierarchy) < 2:
            return None

        parent_code = hierarchy[1]  # First parent

        # Find all cousin codes (codes with same parent)
        cousin_embeddings = []
        cousin_codes = []

        for code, embedding in available_embeddings.items():
            # Check if this code shares the same parent
            if code != novel_code and code.startswith(parent_code):
                # Verify it's at the same hierarchical level as novel_code
                if len(code) == len(novel_code):
                    cousin_embeddings.append(embedding)
                    cousin_codes.append(code)

        if cousin_embeddings:
            # Average all cousin embeddings
            avg_embedding = torch.stack(cousin_embeddings).mean(dim=0)
            logger.info(f"Found {len(cousin_codes)} cousins for {novel_code} (parent: {parent_code}): {cousin_codes[:5]}{'...' if len(cousin_codes) > 5 else ''}")
            return avg_embedding

        logger.debug(f"No cousin embeddings found for {novel_code}")
        return None

    def find_best_parent_embedding(
        self,
        novel_code: str,
        available_embeddings: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Find the best parent embedding for a novel code

        Args:
            novel_code: The code without an embedding
            available_embeddings: Dictionary of available embeddings

        Returns:
            Parent embedding tensor, or None if no parent found
        """
        hierarchy = self.get_code_hierarchy(novel_code)

        # Try each level of hierarchy from most specific to most general
        for parent_code in hierarchy[1:]:  # Skip the original code

            # Try different format variations of parent code
            parent_variations = [
                parent_code,
                parent_code.replace('.', ''),
                f"ICD_{parent_code}",
            ]

            # Add vocab-specific prefixes based on code type
            if self.is_icd10_code(parent_code):
                parent_variations.append(f"ICD10CM_{parent_code}")
            elif self.is_icd9_code(parent_code):
                parent_variations.append(f"ICD9CM_{parent_code}")
            elif self.is_atc_code(parent_code):
                parent_variations.append(f"ATC_{parent_code}")

            for variation in parent_variations:
                if variation in available_embeddings:
                    logger.info(f"Found parent embedding: {novel_code} -> {variation}")
                    return available_embeddings[variation]

        logger.debug(f"No parent embedding found for {novel_code}")
        return None


def integrate_clinvec_with_exmedbert(
    model,
    code_dict,
    clinvec_dir: str,
    vocab_types: List[str] = ["icd9cm", "icd10cm", "phecode"],
    resize_if_needed: bool = True,
    resize_method: str = "auto",
    use_hierarchical_init: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Integrate ClinVec embeddings with ExMed-BERT model

    Args:
        model: ExMed-BERT model instance
        code_dict: Code dictionary from ExMed-BERT
        clinvec_dir: Path to ClinVec dataset directory
        vocab_types: List of vocabulary types to load
        resize_if_needed: Whether to resize embeddings if dimensions don't match
        resize_method: Method for resizing embeddings - "auto", "truncate", "pca",
                      "learned_projection", "pad_smart", "pad_random"
        use_hierarchical_init: Whether to use hierarchical initialization for novel codes
        verbose: Whether to print detailed progress

    Returns:
        Dictionary with loading statistics per vocabulary type
    """
    loader = ClinVecLoader(clinvec_dir)
    hierarchical_matcher = HierarchicalCodeMatcher() if use_hierarchical_init else None
    model_dim = model.config.hidden_size
    stats = {}
    
    logger.info(f"Integrating ClinVec embeddings with ExMed-BERT (model_dim={model_dim})")
    if use_hierarchical_init:
        logger.info("Using hierarchical initialization for novel codes")
    
    # Collect all available embeddings for hierarchical matching
    all_available_embeddings = {}
    
    for vocab_type in vocab_types:
        if verbose:
            print(f"\n=== Loading {vocab_type.upper()} embeddings ===")
        
        # Load embeddings for this vocabulary
        clinvec_embeddings = loader.load_embeddings_by_vocab(vocab_type)
        
        if not clinvec_embeddings:
            stats[vocab_type] = 0
            continue
        
        # Check dimensions and resize if needed
        sample_emb = next(iter(clinvec_embeddings.values()))
        clinvec_dim = sample_emb.shape[0]
        
        if clinvec_dim != model_dim:
            if resize_if_needed:
                if verbose:
                    print(f"Resizing {vocab_type} embeddings: {clinvec_dim} -> {model_dim}")
                clinvec_embeddings = loader.resize_embeddings(clinvec_embeddings, model_dim, resize_method)
            else:
                logger.warning(f"Dimension mismatch for {vocab_type}: {clinvec_dim} vs {model_dim}")
                stats[vocab_type] = 0
                continue
        
        # Add to available embeddings for hierarchical matching
        all_available_embeddings.update(clinvec_embeddings)
        
        # Phase 1: Direct matching - load exact matches
        loaded_count = 0
        matched_codes = set()

        # Debug: Show sample codes for format comparison
        if verbose:
            clinvec_sample = list(clinvec_embeddings.keys())[:10]
            print(f"Sample ClinVec codes: {clinvec_sample}")

            # Debug training vocabulary structure
            print(f"Code dict type: {type(code_dict)}")
            print(f"Code dict attributes: {dir(code_dict)}")

            if hasattr(code_dict, 'stoi'):
                print(f"stoi type: {type(code_dict.stoi)}")
                print(f"stoi length: {len(code_dict.stoi) if code_dict.stoi else 'None/Empty'}")
                if code_dict.stoi:
                    vocab_sample = list(code_dict.stoi.keys())[:10]
                    print(f"Sample training vocab codes (stoi): {vocab_sample}")

            if hasattr(code_dict, 'entity_to_id'):
                print(f"entity_to_id type: {type(code_dict.entity_to_id)}")
                print(f"entity_to_id length: {len(code_dict.entity_to_id) if code_dict.entity_to_id else 'None/Empty'}")
                if code_dict.entity_to_id:
                    vocab_sample = list(code_dict.entity_to_id.keys())[:10]
                    print(f"Sample training vocab codes (entity_to_id): {vocab_sample}")

            if hasattr(code_dict, '__contains__'):
                print(f"Code dict supports 'in' operator")

            if hasattr(code_dict, 'keys'):
                vocab_sample = list(code_dict.keys())[:10] if callable(getattr(code_dict, 'keys', None)) else []
                print(f"Sample training vocab codes (keys): {vocab_sample}")

            # Debug the actual vocabularies
            if hasattr(code_dict, 'vocab'):
                print(f"Vocab length: {len(code_dict.vocab)}")
                print(f"Sample vocab: {list(code_dict.vocab)[:20]}")

            if hasattr(code_dict, 'labels_to_id'):
                print(f"labels_to_id length: {len(code_dict.labels_to_id)}")
                # Look specifically for ICD-10 codes
                icd_codes = [code for code in code_dict.labels_to_id.keys() if isinstance(code, str) and len(code) >= 3 and code[0].isalpha()]
                print(f"Sample ICD-like codes: {icd_codes[:20]}")

            if hasattr(code_dict, 'icd_phewas_map'):
                print(f"icd_phewas_map length: {len(code_dict.icd_phewas_map) if code_dict.icd_phewas_map else 'None'}")
                if code_dict.icd_phewas_map:
                    print(f"Sample ICD-to-Phewas mappings: {list(code_dict.icd_phewas_map.items())[:10]}")

        with torch.no_grad():
            for original_code, embedding in clinvec_embeddings.items():
                # Try different code formats that might exist in ExMed-BERT vocabulary
                possible_codes = [
                    original_code,                    # E.g., "250.0"
                    original_code.replace('.', ''),   # E.g., "2500"
                    f"ICD_{original_code}",          # E.g., "ICD_250.0"
                    f"{vocab_type.upper()}_{original_code}",  # E.g., "ICD9CM_250.0"
                ]
                
                # For ICD codes, also try without decimal
                if '.' in original_code:
                    possible_codes.extend([
                        original_code.replace('.', ''),
                        f"ICD_{original_code.replace('.', '')}"
                    ])
                
                # Check if any variant exists in model vocabulary
                for code_variant in possible_codes:
                    # Check ExMed-BERT CodeDict format first (labels_to_id)
                    if hasattr(code_dict, 'labels_to_id') and code_variant in code_dict.labels_to_id:
                        vocab_idx = code_dict.labels_to_id[code_variant]
                        # Handle different model structures
                        if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                            model.bert.embeddings.code_embeddings.weight[vocab_idx] = embedding
                        elif hasattr(model, 'embeddings'):
                            model.embeddings.code_embeddings.weight[vocab_idx] = embedding
                        else:
                            logger.warning(f"Could not find embeddings in model structure for {code_variant}")
                            continue
                        loaded_count += 1
                        matched_codes.add(code_variant)
                        if verbose and loaded_count <= 5:  # Show first few matches
                            print(f"  + {original_code} -> {code_variant} (idx: {vocab_idx})")
                        break
                    # Fallback: check for stoi attribute (other tokenizer formats)
                    elif hasattr(code_dict, 'stoi') and code_variant in code_dict.stoi:
                        vocab_idx = code_dict.stoi[code_variant]
                        # Handle different model structures
                        if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                            model.bert.embeddings.code_embeddings.weight[vocab_idx] = embedding
                        elif hasattr(model, 'embeddings'):
                            model.embeddings.code_embeddings.weight[vocab_idx] = embedding
                        else:
                            logger.warning(f"Could not find embeddings in model structure for {code_variant}")
                            continue
                        loaded_count += 1
                        matched_codes.add(code_variant)
                        if verbose and loaded_count <= 5:  # Show first few matches
                            print(f"  + {original_code} -> {code_variant} (idx: {vocab_idx})")
                        break
                    # Fallback: check for generic dictionary interface
                    elif hasattr(code_dict, '__contains__') and code_variant in code_dict:
                        try:
                            vocab_idx = code_dict[code_variant]
                            # Handle different model structures
                            if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                                model.bert.embeddings.code_embeddings.weight[vocab_idx] = embedding
                            elif hasattr(model, 'embeddings'):
                                model.embeddings.code_embeddings.weight[vocab_idx] = embedding
                            else:
                                logger.warning(f"Could not find embeddings in model structure for {code_variant}")
                                continue
                            loaded_count += 1
                            matched_codes.add(code_variant)
                            if verbose and loaded_count <= 5:
                                print(f"  + {original_code} -> {code_variant} (idx: {vocab_idx})")
                            break
                        except (KeyError, TypeError):
                            # Dictionary interface failed, continue to next variant
                            continue
        
        stats[vocab_type] = loaded_count
        
        if verbose:
            print(f"  Direct matches: {loaded_count}/{len(clinvec_embeddings)} {vocab_type} embeddings")
            coverage = (loaded_count / len(clinvec_embeddings)) * 100
            print(f"  Direct coverage: {coverage:.1f}%")
    
    # Phase 2: Hierarchical initialization for novel codes
    if use_hierarchical_init and hierarchical_matcher:
        hierarchical_count = 0
        if verbose:
            print(f"\n=== Hierarchical Initialization ===")
        
        with torch.no_grad():
            # Find novel codes in vocabulary that don't have embeddings
            # Check ExMed-BERT CodeDict format first (labels_to_id)
            cousin_count = 0
            parent_count = 0
            if hasattr(code_dict, 'labels_to_id'):
                for vocab_code, vocab_idx in code_dict.labels_to_id.items():
                    if vocab_code not in matched_codes:
                        novel_embedding = None
                        init_method = None

                        # For ATC codes, first try cousin-based initialization
                        if hierarchical_matcher.is_atc_code(vocab_code):
                            cousin_embedding = hierarchical_matcher.find_cousin_average_embedding(
                                vocab_code, all_available_embeddings
                            )
                            if cousin_embedding is not None:
                                # Use cousin average directly (no noise needed)
                                novel_embedding = cousin_embedding
                                init_method = "cousin"
                                cousin_count += 1

                        # If no cousin embedding found, try parent-based initialization
                        if novel_embedding is None:
                            parent_embedding = hierarchical_matcher.find_best_parent_embedding(
                                vocab_code, all_available_embeddings
                            )
                            if parent_embedding is not None:
                                # Add small noise to distinguish from parent
                                noise = torch.normal(0, 0.02, size=parent_embedding.shape)
                                novel_embedding = parent_embedding + noise
                                init_method = "parent"
                                parent_count += 1

                        if novel_embedding is not None:
                            # Handle different model structures
                            if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                                model.bert.embeddings.code_embeddings.weight[vocab_idx] = novel_embedding
                            elif hasattr(model, 'embeddings'):
                                model.embeddings.code_embeddings.weight[vocab_idx] = novel_embedding
                            else:
                                logger.warning(f"Could not find embeddings in model structure for hierarchical init of {vocab_code}")
                                continue
                            hierarchical_count += 1

                            if verbose and hierarchical_count <= 10:
                                print(f"  o {vocab_code} initialized from {init_method}")
            # Fallback for other tokenizer formats
            elif hasattr(code_dict, 'stoi'):
                for vocab_code, vocab_idx in code_dict.stoi.items():
                    if vocab_code not in matched_codes:
                        novel_embedding = None
                        init_method = None

                        # For ATC codes, first try cousin-based initialization
                        if hierarchical_matcher.is_atc_code(vocab_code):
                            cousin_embedding = hierarchical_matcher.find_cousin_average_embedding(
                                vocab_code, all_available_embeddings
                            )
                            if cousin_embedding is not None:
                                novel_embedding = cousin_embedding
                                init_method = "cousin"
                                cousin_count += 1

                        # If no cousin embedding found, try parent-based initialization
                        if novel_embedding is None:
                            parent_embedding = hierarchical_matcher.find_best_parent_embedding(
                                vocab_code, all_available_embeddings
                            )
                            if parent_embedding is not None:
                                # Add small noise to distinguish from parent
                                noise = torch.normal(0, 0.02, size=parent_embedding.shape)
                                novel_embedding = parent_embedding + noise
                                init_method = "parent"
                                parent_count += 1

                        if novel_embedding is not None:
                            # Handle different model structures
                            if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                                model.bert.embeddings.code_embeddings.weight[vocab_idx] = novel_embedding
                            elif hasattr(model, 'embeddings'):
                                model.embeddings.code_embeddings.weight[vocab_idx] = novel_embedding
                            else:
                                logger.warning(f"Could not find embeddings in model structure for hierarchical init of {vocab_code}")
                                continue
                            hierarchical_count += 1

                            if verbose and hierarchical_count <= 10:
                                print(f"  o {vocab_code} initialized from {init_method}")
        
        if verbose:
            print(f"  Hierarchical initializations: {hierarchical_count}")
            if cousin_count > 0:
                print(f"    - Cousin-based (ATC): {cousin_count}")
            if parent_count > 0:
                print(f"    - Parent-based: {parent_count}")

        # Add hierarchical count to stats
        stats['hierarchical_init'] = hierarchical_count
        stats['cousin_init'] = cousin_count
        stats['parent_init'] = parent_count
    
    # Mark embeddings as pre-loaded to prevent re-initialization
    if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
        model.bert.embeddings.code_embeddings._clinvec_loaded = True
    elif hasattr(model, 'embeddings'):
        model.embeddings.code_embeddings._clinvec_loaded = True
    
    total_loaded = sum(stats.values())
    if verbose:
        print(f"\n=== Integration Complete ===")
        print(f"Total embeddings loaded: {total_loaded}")
        for vocab, count in stats.items():
            print(f"  {vocab}: {count}")
    
    logger.info(f"ClinVec integration complete: {total_loaded} embeddings loaded")
    
    return stats


def update_exmedbert_init_weights(model):
    """
    Update ExMed-BERT's _init_weights method to preserve ClinVec embeddings
    Call this AFTER loading ClinVec embeddings
    """
    original_init_weights = model._init_weights
    
    def _init_weights_with_clinvec_preservation(module):
        """Modified init_weights that preserves ClinVec embeddings"""
        if isinstance(module, torch.nn.Embedding):
            # Skip re-initialization if ClinVec embeddings were loaded
            if hasattr(module, '_clinvec_loaded'):
                logger.info("Preserving ClinVec embeddings during weight initialization")
                return
        
        # Call original initialization for other modules
        original_init_weights(module)
    
    # Replace the method
    model._init_weights = _init_weights_with_clinvec_preservation
    logger.info("Updated _init_weights method to preserve ClinVec embeddings")


# Example usage functions
def load_for_exmedbert_pretraining(model, code_dict, clinvec_dir: str):
    """Load ClinVec embeddings for ExMed-BERT pre-training"""
    stats = integrate_clinvec_with_exmedbert(
        model=model,
        code_dict=code_dict,
        clinvec_dir=clinvec_dir,
        vocab_types=["icd9cm", "icd10cm", "phecode", "rxnorm", "atc"],
        resize_if_needed=True,
        verbose=True
    )
    
    # Update initialization method
    update_exmedbert_init_weights(model)
    
    return stats


def load_for_exmedbert_finetuning(model, code_dict, clinvec_dir: str, focus_vocab: str = "icd10cm"):
    """Load specific vocabulary embeddings for fine-tuning"""
    stats = integrate_clinvec_with_exmedbert(
        model=model,
        code_dict=code_dict,
        clinvec_dir=clinvec_dir,
        vocab_types=[focus_vocab],
        resize_if_needed=True,
        use_hierarchical_init=True,
        verbose=True
    )
    
    update_exmedbert_init_weights(model)
    
    return stats


def test_hierarchical_matching():
    """Test function to demonstrate hierarchical code matching"""
    matcher = HierarchicalCodeMatcher()
    
    test_cases = [
        "E11.65",    # Type 2 diabetes with hyperglycemia
        "E11.9",     # Type 2 diabetes without complications
        "250.01",    # Diabetes mellitus without mention of complication, type I
        "I10",       # Essential hypertension
        "Z51.11"     # Encounter for antineoplastic chemotherapy
    ]
    
    print("=== Hierarchical Code Matching Test ===")
    
    for code in test_cases:
        hierarchy = matcher.get_code_hierarchy(code)
        print(f"\n{code}:")
        print(f"  Type: {'ICD-10' if matcher.is_icd10_code(code) else 'ICD-9' if matcher.is_icd9_code(code) else 'Other'}")
        print(f"  Hierarchy: {' → '.join(hierarchy)}")
    
    # Test parent finding with mock embeddings
    mock_embeddings = {
        "E11": torch.randn(128),
        "E11.6": torch.randn(128),
        "250": torch.randn(128),
        "250.0": torch.randn(128)
    }
    
    print(f"\n=== Parent Embedding Test ===")
    print("Available embeddings:", list(mock_embeddings.keys()))
    
    for novel_code in ["E11.65", "E11.321", "250.01", "999.99"]:
        parent_emb = matcher.find_best_parent_embedding(novel_code, mock_embeddings)
        if parent_emb is not None:
            print(f"{novel_code}: Found parent embedding +")
        else:
            print(f"{novel_code}: No parent found x")


if __name__ == "__main__":
    test_hierarchical_matching()