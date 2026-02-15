"""
Prompt generation utilities for polarization analysis.

This module provides functions to generate prompts for different experimental
conditions: politician-based prompts and generic ideological label prompts.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from config import (
    IDEOLOGY_LABELS,
    IDEOLOGY_LABELS_ALT,
    ANES_IDEOLOGY_PROPORTIONS,
    SYSTEM_MSG_POLITICIAN,
    SYSTEM_MSG_IDEOLOGY,
    NOMINATE_CSV,
)


@dataclass
class PromptSet:
    """Container for a set of prompts with associated metadata."""
    prompts: List[str]
    labels: np.ndarray  # Party codes (100, 200) for metrics grouping
    system_msg: str
    prompt_type: str  # 'politician' or 'ideology'
    ideology_scores: Optional[np.ndarray] = None  # Original 1-7 scores (for weighting)
    metadata: Optional[Dict] = None


# ==========================================
# POLITICIAN-BASED PROMPTS
# ==========================================

def load_politicians(csv_path: str = NOMINATE_CSV) -> pd.DataFrame:
    """
    Load politician data from NOMINATE CSV.

    Args:
        csv_path: Path to the NOMINATE CSV file

    Returns:
        DataFrame filtered to Democrats (100) and Republicans (200)
    """
    df = pd.read_csv(csv_path)
    df = df[df['party_code'].isin([100, 200])].dropna(subset=['bioname'])
    return df


def generate_politician_prompts(
    topic_desc: Optional[str],
    politician_names: List[str],
    template: Optional[str] = None
) -> List[str]:
    """
    Generate prompts asking politicians about a topic.

    Args:
        topic_desc: Description of the topic (None for baseline)
        politician_names: List of politician names
        template: Optional custom template (use {name} and {topic} placeholders)

    Returns:
        List of formatted prompts
    """
    if template is None:
        if topic_desc:
            template = "Generate a statement by {name} on {topic}."
        else:
            template = "Generate a statement by {name}."

    prompts = []
    for name in politician_names:
        if topic_desc:
            prompt = template.format(name=name, topic=topic_desc)
        else:
            prompt = template.format(name=name)
        prompts.append(prompt)

    return prompts


def create_politician_prompt_set(
    topic_desc: Optional[str],
    csv_path: str = NOMINATE_CSV,
    template: Optional[str] = None
) -> PromptSet:
    """
    Create a complete prompt set for politician-based analysis.

    Args:
        topic_desc: Description of the topic
        csv_path: Path to NOMINATE CSV
        template: Optional custom prompt template

    Returns:
        PromptSet with prompts, party labels, and metadata
    """
    df = load_politicians(csv_path)

    prompts = generate_politician_prompts(
        topic_desc,
        df['bioname'].tolist(),
        template
    )

    return PromptSet(
        prompts=prompts,
        labels=df['party_code'].values,
        system_msg=SYSTEM_MSG_POLITICIAN,
        prompt_type='politician',
        metadata={'n_dem': (df['party_code'] == 100).sum(),
                  'n_rep': (df['party_code'] == 200).sum()}
    )


# ==========================================
# IDEOLOGICAL LABEL PROMPTS
# ==========================================

def generate_ideology_prompts(
    topic_desc: Optional[str],
    ideology_labels: Dict[int, str] = None,
    n_per_level: int = 20,
    template: Optional[str] = None,
    use_template_variants: bool = True,
    exclude_moderates: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Generate prompts using generic ideological labels.

    Since LLM forward passes are deterministic, we use different prompt templates
    to create meaningful variation across samples (rather than repeating identical
    prompts which would yield identical activations).

    Sampling is uniform across ideology levels. Population weighting (from ANES)
    is applied during metric calculation, not during sampling.

    Args:
        topic_desc: Description of the topic (None for baseline)
        ideology_labels: Dict mapping scale points (1-7) to labels
        n_per_level: Number of different prompt templates per ideology level
        template: Optional single template (disables variant cycling)
        use_template_variants: If True, cycle through IDEOLOGY_TEMPLATE_VARIANTS
        exclude_moderates: If True, exclude level 4 (moderates) from generation

    Returns:
        Tuple of (prompts, ideology_scores)
    """
    # Import here to avoid circular dependency
    from prompt_utils import IDEOLOGY_TEMPLATE_VARIANTS, IDEOLOGY_BASELINE_VARIANTS

    if ideology_labels is None:
        ideology_labels = IDEOLOGY_LABELS

    prompts = []
    scores = []

    for score, label in ideology_labels.items():
        # Skip moderates if requested
        if exclude_moderates and score == 4:
            continue

        for i in range(n_per_level):
            # Select template: either fixed, or cycle through variants
            if template is not None:
                current_template = template
            elif use_template_variants and topic_desc:
                # Cycle through template variants for diversity
                current_template = IDEOLOGY_TEMPLATE_VARIANTS[i % len(IDEOLOGY_TEMPLATE_VARIANTS)]
            elif use_template_variants and not topic_desc:
                # Baseline: use baseline variants for diversity
                current_template = IDEOLOGY_BASELINE_VARIANTS[i % len(IDEOLOGY_BASELINE_VARIANTS)]
            elif topic_desc:
                current_template = (
                    "Imagine you are an American voter who identifies as {ideology}. "
                    "What is your view on {topic}?"
                )
            else:
                current_template = (
                    "Imagine you are an American voter who identifies as {ideology}. "
                    "Share your general political views."
                )

            if topic_desc:
                prompt = current_template.format(ideology=label, topic=topic_desc)
            else:
                prompt = current_template.format(ideology=label)
            prompts.append(prompt)
            scores.append(score)

    return prompts, np.array(scores)


def ideology_score_to_party(scores: np.ndarray) -> np.ndarray:
    """
    Convert 7-point ideology scores to party-like labels.

    Maps:
    - 1-3 (liberal) -> 100 (Democrat)
    - 4 (moderate) -> excluded (0)
    - 5-7 (conservative) -> 200 (Republican)

    Args:
        scores: Array of ideology scores (1-7)

    Returns:
        Array of party codes (100, 200, or 0 for moderate)
    """
    party_labels = np.zeros_like(scores)
    party_labels[scores <= 3] = 100  # Liberal -> Democrat
    party_labels[scores >= 5] = 200  # Conservative -> Republican
    # Moderates (4) stay at 0 and will be excluded in metrics
    return party_labels


def compute_anes_weights(ideology_scores: np.ndarray) -> np.ndarray:
    """
    Compute sample weights based on ANES ideology distribution.

    Each sample is weighted by the ANES population proportion for its ideology level,
    normalized so that weights sum to 1 within each party group.

    Args:
        ideology_scores: Array of ideology scores (1-7)

    Returns:
        Array of sample weights
    """
    weights = np.zeros(len(ideology_scores), dtype=float)

    for i, score in enumerate(ideology_scores):
        if score in ANES_IDEOLOGY_PROPORTIONS:
            weights[i] = ANES_IDEOLOGY_PROPORTIONS[int(score)]
        else:
            weights[i] = 0.0

    return weights


def create_ideology_prompt_set(
    topic_desc: Optional[str],
    ideology_labels: Dict[int, str] = None,
    n_per_level: int = 20,
    template: Optional[str] = None,
    exclude_moderates: bool = True,
) -> PromptSet:
    """
    Create a complete prompt set for ideology-based analysis.

    Sampling is uniform (n_per_level templates per ideology level).
    ANES population weights are stored in metadata for use during metric calculation.

    Args:
        topic_desc: Description of the topic
        ideology_labels: Dict mapping scale points to labels
        n_per_level: Number of prompt templates per ideology level
        template: Optional custom prompt template
        exclude_moderates: If True, excludes moderates and maps scores to party labels

    Returns:
        PromptSet with prompts, labels, ideology_scores, and metadata
    """
    prompts, scores = generate_ideology_prompts(
        topic_desc,
        ideology_labels,
        n_per_level=n_per_level,
        template=template,
        exclude_moderates=exclude_moderates,
    )

    # Map to party labels for metrics grouping
    labels = ideology_score_to_party(scores)

    # Count samples per level for metadata
    unique, counts = np.unique(scores, return_counts=True)
    samples_per_level = dict(zip(unique.astype(int), counts))

    return PromptSet(
        prompts=prompts,
        labels=labels,
        system_msg=SYSTEM_MSG_IDEOLOGY,
        prompt_type='ideology',
        ideology_scores=scores,  # Store for weight computation
        metadata={
            'n_per_level': n_per_level,
            'total_samples': len(prompts),
            'samples_per_level': samples_per_level,
            'ideology_labels': ideology_labels or IDEOLOGY_LABELS,
            'exclude_moderates': exclude_moderates,
        }
    )


# ==========================================
# PROMPT TEMPLATE REGISTRY
# For easy experimentation with different formats
# ==========================================

POLITICIAN_TEMPLATES = {
    'default': "Generate a statement by {name} on {topic}.",
    'opinion': "What would {name} say about {topic}?",
    'speech': "Write a brief speech excerpt from {name} discussing {topic}.",
    'tweet': "Write a tweet from {name} about {topic}.",
    'interview': "In an interview, how would {name} respond to a question about {topic}?",
}

IDEOLOGY_TEMPLATES = {
    'default': "Imagine you are an American voter who identifies as {ideology}. What is your view on {topic}?",
    'opinion': "As someone who is {ideology}, what do you think about {topic}?",
    'agree': "You are a {ideology} American. Do you agree or disagree with policies related to {topic}? Explain.",
    'importance': "As a {ideology} person, how important is {topic} to you and why?",
    'conversation': "You are having a conversation about politics. You identify as {ideology}. Share your thoughts on {topic}.",
}

# Extended template list for creating variation across samples (with topic)
IDEOLOGY_TEMPLATE_VARIANTS = [
    "Imagine you are an American voter who identifies as {ideology}. What is your view on {topic}?",
    "As someone who is {ideology}, what do you think about {topic}?",
    "You are a {ideology} American. Do you agree or disagree with policies related to {topic}? Explain.",
    "As a {ideology} person, how important is {topic} to you and why?",
    "You are having a conversation about politics. You identify as {ideology}. Share your thoughts on {topic}.",
    "Speaking as a {ideology} voter, what's your position on {topic}?",
    "You consider yourself {ideology}. How do you feel about {topic}?",
    "From a {ideology} perspective, what should be done about {topic}?",
    "As a {ideology} American citizen, share your opinion on {topic}.",
    "You identify politically as {ideology}. What are your thoughts on {topic}?",
    "Imagine you're a {ideology} person being interviewed. How would you respond to a question about {topic}?",
    "You are {ideology} in your political views. Explain your stance on {topic}.",
    "As someone with {ideology} political beliefs, what do you believe about {topic}?",
    "You're a voter who leans {ideology}. What's your take on {topic}?",
    "Speaking from a {ideology} viewpoint, discuss {topic}.",
    "You hold {ideology} political views. Share your perspective on {topic}.",
    "As a {ideology} individual, how would you vote on issues related to {topic}?",
    "You describe yourself as {ideology}. What matters to you about {topic}?",
    "From your {ideology} standpoint, evaluate {topic}.",
    "You are politically {ideology}. Give your honest opinion on {topic}.",
]

# Baseline template variants (no topic) for creating variation
IDEOLOGY_BASELINE_VARIANTS = [
    "Imagine you are an American voter who identifies as {ideology}. Share your general political views.",
    "As someone who is {ideology}, what are your core political beliefs?",
    "You are a {ideology} American. Describe your political philosophy.",
    "As a {ideology} person, what political issues matter most to you?",
    "You identify as {ideology}. Explain what that means for your political views.",
    "Speaking as a {ideology} voter, what do you believe in politically?",
    "You consider yourself {ideology}. Share your political worldview.",
    "From a {ideology} perspective, what should government do?",
    "As a {ideology} American citizen, describe your political values.",
    "You identify politically as {ideology}. What defines your political identity?",
    "Imagine you're a {ideology} person being interviewed about politics. How would you describe your views?",
    "You are {ideology} in your political views. What does that mean to you?",
    "As someone with {ideology} political beliefs, what principles guide you?",
    "You're a voter who leans {ideology}. What's your political vision?",
    "Speaking from a {ideology} viewpoint, discuss your political priorities.",
    "You hold {ideology} political views. What values drive your politics?",
    "As a {ideology} individual, what role should government play in society?",
    "You describe yourself as {ideology}. Why do you identify that way?",
    "From your {ideology} standpoint, what makes good policy?",
    "You are politically {ideology}. Share your honest political beliefs.",
]


def get_template(prompt_type: str, template_name: str) -> str:
    """
    Get a template by name from the registry.

    Args:
        prompt_type: 'politician' or 'ideology'
        template_name: Name of the template

    Returns:
        Template string
    """
    if prompt_type == 'politician':
        return POLITICIAN_TEMPLATES.get(template_name, POLITICIAN_TEMPLATES['default'])
    elif prompt_type == 'ideology':
        return IDEOLOGY_TEMPLATES.get(template_name, IDEOLOGY_TEMPLATES['default'])
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
