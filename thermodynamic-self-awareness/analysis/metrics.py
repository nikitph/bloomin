"""
Analysis and Metrics Module

Implements statistical tests, metrics computation, and visualization
for experimental results.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def compute_free_energy_trend(
    free_energy_series: List[float],
    alpha: float = 0.01
) -> Dict:
    """
    Test for significant downward trend in free energy.
    
    Uses Mann-Kendall test for monotonic trend.
    
    Args:
        free_energy_series: Time series of free energy values
        alpha: Significance level
        
    Returns:
        Dictionary with trend statistics
    """
    n = len(free_energy_series)
    if n < 3:
        return {'significant': False, 'p_value': 1.0, 'slope': 0.0}
    
    # Linear regression for slope
    x = np.arange(n)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, free_energy_series)
    
    # Mann-Kendall test
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(free_energy_series[j] - free_energy_series[i])
    
    # Variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Z-score
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Two-tailed p-value
    mk_p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'significant': (slope < 0 and p_value < alpha),
        'slope': slope,
        'p_value': p_value,
        'mann_kendall_p': mk_p_value,
        'r_squared': r_value ** 2
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data
        statistic_fn: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        (statistic, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    
    # Original statistic
    original_stat = statistic_fn(data)
    
    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return original_stat, lower, upper


def compare_discovery_rates(
    experimental_discoveries: List[bool],
    control_discoveries: List[bool],
    alpha: float = 0.01
) -> Dict:
    """
    Compare rule discovery rates between experimental and control groups.
    
    Uses Fisher's exact test.
    
    Args:
        experimental_discoveries: List of discovery outcomes (True/False) for experimental group
        control_discoveries: List of discovery outcomes for control group
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Contingency table
    exp_success = sum(experimental_discoveries)
    exp_failure = len(experimental_discoveries) - exp_success
    ctrl_success = sum(control_discoveries)
    ctrl_failure = len(control_discoveries) - ctrl_success
    
    table = [[exp_success, exp_failure],
             [ctrl_success, ctrl_failure]]
    
    # Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact(table)
    
    # Discovery rates
    exp_rate = exp_success / len(experimental_discoveries) if experimental_discoveries else 0
    ctrl_rate = ctrl_success / len(control_discoveries) if control_discoveries else 0
    
    return {
        'experimental_rate': exp_rate,
        'control_rate': ctrl_rate,
        'odds_ratio': odds_ratio,
        'p_value': p_value,
        'significant': p_value < alpha,
        'n_experimental': len(experimental_discoveries),
        'n_control': len(control_discoveries)
    }


def evaluate_self_healing(
    retrieval_ranks_before: np.ndarray,
    retrieval_ranks_after: np.ndarray,
    threshold: float = 0.95
) -> Dict:
    """
    Evaluate self-healing via retrieval rank correlation.
    
    Args:
        retrieval_ranks_before: Retrieval ranks before corruption
        retrieval_ranks_after: Retrieval ranks after healing
        threshold: Success threshold for correlation
        
    Returns:
        Dictionary with healing metrics
    """
    # Spearman correlation
    correlation, p_value = spearmanr(retrieval_ranks_before, retrieval_ranks_after)
    
    # Success
    success = correlation >= threshold
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'success': success,
        'threshold': threshold
    }


def plot_free_energy_comparison(
    results_dict: Dict[str, List[float]],
    output_path: str,
    title: str = "Free Energy Comparison"
):
    """
    Plot free energy trajectories for multiple conditions.
    
    Args:
        results_dict: Dictionary mapping condition name to free energy series
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for condition, free_energy in results_dict.items():
        plt.plot(free_energy, label=condition, alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Free Energy F')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_kl_heatmap(
    kl_matrix: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "KL Divergence Matrix"
):
    """
    Plot heatmap of KL divergence matrix.
    
    Args:
        kl_matrix: KL divergence matrix
        labels: Labels for rows/columns
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(8, 7))
    
    sns.heatmap(
        kl_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'KL Divergence'}
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_report(
    results: Dict,
    output_path: str
):
    """
    Generate markdown summary report.
    
    Args:
        results: Results dictionary
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("# Thermodynamic Self-Awareness Experiment Results\n\n")
        
        f.write("## Free Energy Analysis\n\n")
        if 'free_energy_trend' in results:
            trend = results['free_energy_trend']
            f.write(f"- **Slope**: {trend['slope']:.6f}\n")
            f.write(f"- **P-value**: {trend['p_value']:.4f}\n")
            f.write(f"- **R²**: {trend['r_squared']:.4f}\n")
            f.write(f"- **Significant Downward Trend**: {'✓ Yes' if trend['significant'] else '✗ No'}\n\n")
        
        f.write("## Rule Discovery\n\n")
        if 'rule_discovery_epoch' in results:
            epoch = results['rule_discovery_epoch']
            if epoch is not None:
                f.write(f"- **Discovery Epoch**: {epoch}\n")
                f.write(f"- **Status**: ✓ Rule discovered\n\n")
            else:
                f.write("- **Status**: ✗ Rule not discovered\n\n")
        
        f.write("## System Statistics\n\n")
        if 'total_contradictions' in results:
            f.write(f"- **Total Contradictions**: {results['total_contradictions']}\n")
        if 'total_ricci_updates' in results:
            f.write(f"- **Total Ricci Updates**: {results['total_ricci_updates']}\n")
        
        f.write("\n---\n\n")
        f.write("*Generated by Thermodynamic Self-Awareness Framework*\n")
    
    print(f"Summary report saved to {output_path}")
