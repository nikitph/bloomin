"""Analysis module initialization"""
from .metrics import (
    compute_free_energy_trend,
    bootstrap_confidence_interval,
    compare_discovery_rates,
    evaluate_self_healing,
    plot_free_energy_comparison,
    plot_kl_heatmap,
    generate_summary_report
)

__all__ = [
    'compute_free_energy_trend',
    'bootstrap_confidence_interval',
    'compare_discovery_rates',
    'evaluate_self_healing',
    'plot_free_energy_comparison',
    'plot_kl_heatmap',
    'generate_summary_report'
]
