"""
Shared utilities for overnight polarization experiments.

This module contains common functions used across all 8 experiments:
- Model configurations
- Topic loading and sampling
- GSS data loading
- Statistical functions (PCA, Mahalanobis, etc.)
- Plotting helpers
"""

import os
import sys
import random
import pickle
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_utils import load_model, extract_heads_batched, get_model_info
from prompt_utils import load_politicians, generate_politician_prompts
from run_gss_pca import load_topics_from_csv

# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = "/project/jevans/maxzhuyt/models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
TOPIC_LISTS_DIR = Path(__file__).parent / "topic_lists"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
TOPIC_LISTS_DIR.mkdir(exist_ok=True)

MODEL_FAMILIES = {
    "Qwen3-4B": {
        "base": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Base",
            "type": "base",
            "batch_size": 180
        },
        "instruct": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507",
            "type": "instruct",
            "batch_size": 180
        },
        "reasoning": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Thinking-2507",
            "type": "reasoning",
            "batch_size": 180
        },
    },
    "Llama-3.1-8B": {
        "base": {
            "path": f"{MODELS_DIR}/Meta-Llama-3.1-8B",
            "type": "base",
            "batch_size": 150
        },
        "instruct": {
            "path": f"{MODELS_DIR}/Meta-Llama-3.1-8B-Instruct",
            "type": "instruct",
            "batch_size": 150
        },
        "reasoning": {
            "path": f"{MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B",
            "type": "reasoning",
            "batch_size": 150
        },
    },
    "Gemma-2-9b": {
        "base": {
            "path": f"{MODELS_DIR}/gemma-2-9b",
            "type": "base",
            "batch_size": 160
        },
        "instruct": {
            "path": f"{MODELS_DIR}/gemma-2-9b-it",
            "type": "instruct",
            "batch_size": 160
        },
    },
}

# =============================================================================
# Reproducibility
# =============================================================================

def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: This doesn't guarantee perfect reproducibility with CUDA operations,
    # but it helps. For perfect reproducibility, would need:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # But this significantly slows down training, so we skip for experiments.

# =============================================================================
# Topic Loading and Sampling
# =============================================================================

def load_all_topics() -> Tuple[Dict, Dict]:
    """
    Load all public and private topics from CSV files.

    Returns:
        public_topics: Dict mapping topic names to descriptions
        private_topics: Dict mapping topic names to descriptions
    """
    # Use absolute path to question_lists at gss_polarization root
    question_dir = Path("/project/jevans/maxzhuyt/gss_polarization/question_lists")

    # load_topics_from_csv returns (topics_dict, df) - we only need the dict
    public_topics, _ = load_topics_from_csv(
        str(question_dir / "public_issues.csv")
    )
    private_topics, _ = load_topics_from_csv(
        str(question_dir / "private_life.csv")
    )

    return public_topics, private_topics

def sample_topics(
    public_topics: Dict,
    private_topics: Dict,
    n_public: int,
    n_private: int,
    seed: int,
    exclude: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """
    Sample random subsets of topics with fixed seed.

    Args:
        public_topics: All public topics
        private_topics: All private topics
        n_public: Number of public topics to sample
        n_private: Number of private topics to sample
        seed: Random seed
        exclude: List of topic names to exclude (e.g., excluded topics from run_gss_pca)

    Returns:
        sampled_public: Sampled public topics
        sampled_private: Sampled private topics
    """
    rng = random.Random(seed)

    if exclude:
        public_topics = {k: v for k, v in public_topics.items() if k not in exclude}
        private_topics = {k: v for k, v in private_topics.items() if k not in exclude}

    public_keys = rng.sample(list(public_topics.keys()), min(n_public, len(public_topics)))
    private_keys = rng.sample(list(private_topics.keys()), min(n_private, len(private_topics)))

    sampled_public = {k: public_topics[k] for k in public_keys}
    sampled_private = {k: private_topics[k] for k in private_keys}

    return sampled_public, sampled_private

def get_excluded_topics() -> List[str]:
    """Get list of excluded topics (same as run_gss_pca.py)."""
    return [
        # Public issues with known measurement issues
        "hubbywk1", "racdif1", "racdif2", "racdif3", "racdif4",
        "workwhts", "wlthwhts", "intlwhts",
        # Private life with known measurement issues
        "reborn", "marwht", "helpful", "helpfulnv", "helpfulv"
    ]

# =============================================================================
# GSS Data Loading
# =============================================================================

def load_polarization_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load aggregated GSS polarization statistics from CSV.

    Args:
        csv_path: Path to CSV file (default: combines public + private polarization data)

    Returns:
        DataFrame with columns: variable, area, polarization, mean_dem, mean_rep, n_dem, n_rep, etc.
    """
    if csv_path is None:
        # Load both public and private polarization data
        public_pol = pd.read_csv(DATA_DIR / "polarization" / "public_issues_polarization.csv")
        private_pol = pd.read_csv(DATA_DIR / "polarization" / "private_life_polarization.csv")

        # Combine
        df = pd.concat([public_pol, private_pol], ignore_index=True)
    else:
        df = pd.read_csv(csv_path)

    return df

def compute_gss_statistics(
    df: pd.DataFrame,
    topic_name: str,
    year_range: Optional[Tuple[int, int]] = None
) -> Dict[str, float]:
    """
    Get GSS-based statistics for a topic from aggregated polarization data.

    Args:
        df: GSS polarization data (aggregated)
        topic_name: Variable name (e.g., 'abany')
        year_range: Not used (aggregated data doesn't have year breakdown)

    Returns:
        Dictionary with:
        - mean_D: Democrat mean
        - mean_R: Republican mean
        - polarization: Polarization score
        - n_D: Democrat sample size
        - n_R: Republican sample size
        - n_total: Total sample size
    """
    topic_row = df[df['variable'] == topic_name]

    if len(topic_row) == 0:
        return None

    row = topic_row.iloc[0]

    # Check minimum sample size
    if row['n_dem'] < 10 or row['n_rep'] < 10:
        return None

    return {
        'mean_D': row['mean_dem'],
        'mean_R': row['mean_rep'],
        'polarization': row['polarization'],
        'n_D': row['n_dem'],
        'n_R': row['n_rep'],
        'n_total': row['n_total'],
    }

# =============================================================================
# Statistical Functions
# =============================================================================

def compute_pca_and_distance(
    activations: np.ndarray,
    party_labels: np.ndarray,
    pca_dim: int = 15
) -> Dict[str, Any]:
    """
    Compute PCA and Mahalanobis distance between Democrat and Republican centroids.

    Args:
        activations: Activation array. Can be:
            - 2D (n_samples, n_features)
            - 4D (n_samples, n_layers, n_heads, head_dim) - will be flattened to 2D
        party_labels: (n_samples,) array of party labels (0=Democrat, 1=Republican)
        pca_dim: Number of PCA dimensions

    Returns:
        Dictionary with:
        - pca_activations: (n_samples, pca_dim) transformed activations
        - variance_explained: Variance explained by first PC
        - mahalanobis_dist: Mahalanobis distance between centroids
        - centroid_D: Democrat centroid in PCA space
        - centroid_R: Republican centroid in PCA space
    """
    # Flatten high-dimensional activations to 2D
    if activations.ndim > 2:
        n_samples = activations.shape[0]
        activations = activations.reshape(n_samples, -1)

    # Standardize
    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(activations)

    # PCA
    pca = PCA(n_components=min(pca_dim, activations.shape[1]))
    pca_activations = pca.fit_transform(activations_scaled)

    # Centroids
    dem_mask = party_labels == 0
    rep_mask = party_labels == 1

    centroid_D = np.mean(pca_activations[dem_mask], axis=0)
    centroid_R = np.mean(pca_activations[rep_mask], axis=0)

    # Pooled covariance matrix
    cov_D = np.cov(pca_activations[dem_mask].T)
    cov_R = np.cov(pca_activations[rep_mask].T)
    n_D = np.sum(dem_mask)
    n_R = np.sum(rep_mask)
    pooled_cov = ((n_D - 1) * cov_D + (n_R - 1) * cov_R) / (n_D + n_R - 2)

    # Add small regularization for numerical stability
    pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6

    # Mahalanobis distance
    try:
        mahal_dist = mahalanobis(centroid_D, centroid_R, np.linalg.inv(pooled_cov))
    except np.linalg.LinAlgError:
        # Fallback to Euclidean if covariance is singular
        mahal_dist = np.linalg.norm(centroid_D - centroid_R)

    return {
        'pca_activations': pca_activations,
        'variance_explained': pca.explained_variance_ratio_[0],  # First PC only
        'mahalanobis_dist': mahal_dist,
        'centroid_D': centroid_D,
        'centroid_R': centroid_R,
        'pca_object': pca,
    }

def compute_overlap_coefficient(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """
    Compute overlap coefficient between two distributions using KDE.

    Args:
        dist1: First distribution (e.g., Democrat activations)
        dist2: Second distribution (e.g., Republican activations)

    Returns:
        Overlap coefficient (0 = no overlap, 1 = complete overlap)
    """
    from scipy.stats import gaussian_kde

    # Ensure 1D
    dist1 = np.asarray(dist1).flatten()
    dist2 = np.asarray(dist2).flatten()

    # Create KDEs
    kde1 = gaussian_kde(dist1, bw_method='scott')
    kde2 = gaussian_kde(dist2, bw_method='scott')

    # Evaluation range
    min_val = min(dist1.min(), dist2.min())
    max_val = max(dist1.max(), dist2.max())
    x = np.linspace(min_val, max_val, 1000)

    # Compute PDFs
    pdf1 = kde1(x)
    pdf2 = kde2(x)

    # Overlap = integral of min(pdf1, pdf2)
    overlap = np.trapz(np.minimum(pdf1, pdf2), x)

    return overlap

def compute_bimodality_coefficient(data: np.ndarray) -> float:
    """
    Compute bimodality coefficient.

    BC = (skewness^2 + 1) / (kurtosis + 3)
    BC > 0.55 suggests bimodal distribution.

    Args:
        data: 1D array

    Returns:
        Bimodality coefficient
    """
    from scipy.stats import skew, kurtosis

    data = np.asarray(data).flatten()

    skewness = skew(data)
    kurt = kurtosis(data, fisher=True)  # Excess kurtosis (normal = 0)

    bc = (skewness**2 + 1) / (kurt + 3)

    return bc

# =============================================================================
# Model Running Helpers
# =============================================================================

def run_model_on_topics(
    model_path: str,
    model_name: str,
    model_type: str,
    topics: Dict[str, str],
    category: str,
    politician_names: List[str],
    batch_size: int,
    max_length: int = 128,
    system_override: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Run a single model on a set of topics and extract activations.

    Mirrors the exact logic from run_model_comparison.py for reliability.

    Args:
        model_path: Path to model
        model_name: Model name for logging
        model_type: 'base', 'instruct', or 'reasoning'
        topics: Dict mapping topic names to descriptions
        category: 'public_issues' or 'private_life'
        politician_names: List of politician name strings
        batch_size: Batch size for extraction
        max_length: Max sequence length
        system_override: Optional system message override (e.g., "/no_think")

    Returns:
        Dictionary mapping topic_name to (n_politicians, n_heads) activation matrix
    """
    from config import SYSTEM_MSG_POLITICIAN
    from prompt_utils import generate_politician_prompts, POLITICIAN_TEMPLATES

    # Same template mappings as run_model_comparison.py
    BASE_TEMPLATES = {
        "public_issues": "{name} makes a statement on {topic}:",
        "private_life":  "When asked about {topic}, {name} says",
    }
    INSTRUCT_TEMPLATE_KEYS = {
        "public_issues": "default",
        "private_life":  "opinion",
    }

    print(f"\n{'='*80}")
    print(f"Running {model_name} ({model_type}) on {len(topics)} {category} topics")
    print(f"{'='*80}\n")

    # Load model (identical to run_model_comparison.py)
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path)

    # Disable chat template for base models
    if model_type == "base":
        tokenizer.chat_template = None
        print("Disabled chat template for base model")

    # Extract activations for each topic
    results = {}

    for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
        print(f"\n[{topic_idx}/{len(topics)}] Processing topic: {topic_name}")

        # Generate prompts + system_msg (same logic as run_model_comparison.py)
        if model_type == "base":
            template = BASE_TEMPLATES[category]
            prompts = [template.format(name=name, topic=topic_desc)
                       for name in politician_names]
            system_msg = ""
        else:
            template_key = INSTRUCT_TEMPLATE_KEYS[category]
            template = POLITICIAN_TEMPLATES[template_key]
            prompts = generate_politician_prompts(
                topic_desc, politician_names, template=template
            )
            system_msg = (system_override if system_override is not None
                         else SYSTEM_MSG_POLITICIAN)

        # Extract activations (same call as run_model_comparison.py)
        activations = extract_heads_batched(
            model, tokenizer, prompts, system_msg,
            batch_size=batch_size, max_length=max_length,
        )  # Shape: (n_politicians, n_heads)

        results[topic_name] = activations

        print(f"  Extracted activations: {activations.shape}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # Unload model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nFinished {model_name}")

    return results

# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(data: Any, experiment_name: str, checkpoint_name: str):
    """
    Save experiment checkpoint.

    Args:
        data: Data to save (dictionary, dataframe, etc.)
        experiment_name: Experiment identifier (e.g., 'exp1', 'exp2')
        checkpoint_name: Checkpoint identifier (e.g., 'Qwen3-4B_base', 'final')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{checkpoint_name}_{timestamp}.pkl"
    filepath = RESULTS_DIR / filename

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved checkpoint: {filepath}")

    return filepath

def load_checkpoint(filepath: Path) -> Any:
    """Load experiment checkpoint."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# =============================================================================
# Plotting Helpers
# =============================================================================

def setup_plot_style():
    """Set up consistent plotting style for all experiments."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

def save_figure(fig: plt.Figure, experiment_name: str, plot_name: str):
    """
    Save figure with consistent naming.

    Args:
        fig: Matplotlib figure
        experiment_name: Experiment identifier (e.g., 'exp1')
        plot_name: Plot identifier (e.g., 'pca_variance')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{plot_name}_{timestamp}.png"
    filepath = RESULTS_DIR / filename

    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filepath}")

    return filepath
