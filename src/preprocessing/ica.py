"""
FastICA-based artifact removal for EEG signals.

Uses sklearn's FastICA to decompose EEG into independent components.
Removes the N components with highest variance (typically muscle/ocular artifacts).
"""
import numpy as np
from sklearn.decomposition import FastICA

import config as cfg


def apply_ica(signals: np.ndarray, n_remove: int = None,
              max_iter: int = None, random_state: int = None) -> np.ndarray:
    """Apply ICA and remove the top-N highest-variance components.

    Args:
        signals: (n_channels, n_samples) preprocessed EEG.
        n_remove: Number of components to remove (default from config).
        max_iter: Maximum ICA iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Cleaned signals with same shape as input.
    """
    n_remove = n_remove or cfg.N_ICA_REMOVE
    max_iter = max_iter or cfg.ICA_MAX_ITER
    random_state = random_state or cfg.ICA_RANDOM_STATE
    n_comp = signals.shape[0]

    if n_comp <= n_remove:
        return signals

    try:
        ica = FastICA(n_components=n_comp, max_iter=max_iter, random_state=random_state)
        components = ica.fit_transform(signals.T).T  # (n_comp, n_samples)

        # Remove components with highest variance (same logic as MATLAB)
        variances = np.var(components, axis=1)
        sorted_idx = np.argsort(variances)[::-1]
        components[sorted_idx[:n_remove], :] = 0

        # Project back to channel space
        cleaned = ica.mixing_.dot(components)
        return cleaned

    except Exception as e:
        print(f"    [WARNING] ICA failed: {e}. Continuing without ICA.")
        return signals
