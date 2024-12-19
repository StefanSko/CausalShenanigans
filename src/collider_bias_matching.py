import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 2000

# Generate two features that influence matching
u = np.random.normal(0, 1, n_samples)  # Primary feature (e.g., content similarity)
v = np.random.normal(0, 1, n_samples)  # Secondary feature (e.g., contextual relevance)

# Convert likelihood to probability using sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Create a TRUE effect where both features influence the outcome
true_match_logits = 0.2 * v + 0.8 * u + np.random.normal(0, 1, n_samples)
true_match_probs = sigmoid(true_match_logits)

# Generate actual binary outcomes
true_matches = np.random.binomial(n=1, p=true_match_probs)

# Selection mechanism based on feature combination
S = np.where(0.5 * u + 0.5 * v > 0.5, 1, 0)  # Item is shown if combination exceeds threshold

# Calculate correlations in full dataset vs selected dataset
corr_full = stats.pearsonr(v, true_match_probs)
corr_selected = stats.pearsonr(v[S==1], true_match_probs[S==1])

# Simulate intervention: New selection mechanism with different weights
S_intervention = np.where(0.9 * v + 0.1 * u > 0.5, 1, 0)  # Modified feature weights

# Calculate correlations for intervention dataset
corr_intervention = stats.pearsonr(v[S_intervention==1], true_matches[S_intervention==1])


# Calculate business metrics
true_matches = np.random.binomial(n=1, p=true_match_probs[S==1])
adjusted_matches = np.random.binomial(n=1, p=true_match_probs[S_intervention==1])
diversity_original = np.std(u[S==1])
diversity_intervention = np.std(u[S_intervention==1])

# Extend visualization to show intervention effects
plt.figure(figsize=(15, 10))

# Original plots (first row)
plt.subplot(2, 3, 1)
plt.scatter(v, true_match_logits, alpha=0.5)
plt.xlabel('Feature V')
plt.ylabel('Match Probability')
plt.title(f'True Relationship\nCorrelation: {corr_full[0]:.2f}')

plt.subplot(2, 3, 2)
plt.scatter(v, u, c=S, cmap='viridis', alpha=0.5)
plt.colorbar(label='Selected (Shown)')
plt.xlabel('Feature V')
plt.ylabel('Feature U')
plt.title('Original Selection Mechanism')

plt.subplot(2, 3, 3)
plt.scatter(v[S==1], true_match_logits[S==1], alpha=0.5)
plt.xlabel('Feature V')
plt.ylabel('Match Probability')
plt.title(f'Original Observed Relationship\nCorrelation: {corr_selected[0]:.2f}')

# Intervention plots (second row)
plt.subplot(2, 3, 5)
plt.scatter(v, u, c=S_intervention, cmap='viridis', alpha=0.5)
plt.colorbar(label='Selected (Shown)')
plt.xlabel('Feature V')
plt.ylabel('Feature U')
plt.title('Intervention Selection Mechanism\n(Heavy Feature V Weight)')

plt.subplot(2, 3, 6)
plt.scatter(v[S_intervention==1], true_match_logits[S_intervention==1], alpha=0.5)
plt.xlabel('Feature V')
plt.ylabel('Match Probability')
plt.title(f'Post-Intervention Relationship\nCorrelation: {corr_intervention[0]:.2f}')

plt.tight_layout()
plt.show()

# Print comprehensive results
print("\nCorrelation Analysis:")
print(f"True correlation: {corr_full[0]:.3f}")
print(f"Original observed correlation: {corr_selected[0]:.3f}")
print(f"Post-intervention correlation: {corr_intervention[0]:.3f}")

print("\nBusiness Metrics:")
print(f"Estimated num. matches (original): {np.sum(true_matches):.3f}")
print(f"Estimated num. matches (intervention): {np.sum(adjusted_matches):.3f}")
print(f"Feature diversity (original): {diversity_original:.3f}")
print(f"Feature diversity (intervention): {diversity_intervention:.3f}")