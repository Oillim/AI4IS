import numpy as np

def _apply_ldp(weights, epsilon):
    """Applies Laplace noise for Local Differential Privacy."""
    noisy_weights = []
    for w in weights:
        sensitivity = np.max(w) - np.min(w)
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, w.shape)
        noisy_weights.append(w + noise)
    return noisy_weights