import matplotlib.pyplot as plt
import numpy as np

def plot_siddhi_curves(results):
    scales = [r[0]['depth'] * r[0]['recurrence'] for r in results]
    mean_responses = [r[1]['mean_response'] for r in results]
    tail_sensitivities = [r[1]['tail_sensitivity'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(scales, mean_responses, marker='o', label='Mean Weak Signal Response')
    plt.title('Siddhi Emergence: Sensitivity vs. Model Scale')
    plt.xlabel('Scale (Depth * Recurrence)')
    plt.ylabel('Mean Response')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(scales, tail_sensitivities, marker='s', color='orange', label='99th Percentile Sensitivity')
    plt.xlabel('Scale (Depth * Recurrence)')
    plt.ylabel('Tail Sensitivity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('siddhi_onset_curves.png')
    print("Siddhi onset curves saved to siddhi_onset_curves.png")

if __name__ == "__main__":
    # Example usage with dummy data
    dummy_results = [
        ({'depth': 1, 'recurrence': 1}, {'mean_response': 0.001, 'tail_sensitivity': 0.005}),
        ({'depth': 5, 'recurrence': 2}, {'mean_response': 0.002, 'tail_sensitivity': 0.012}),
        ({'depth': 10, 'recurrence': 5}, {'mean_response': 0.005, 'tail_sensitivity': 0.045}),
        ({'depth': 20, 'recurrence': 10}, {'mean_response': 0.015, 'tail_sensitivity': 0.082}),
    ]
    plot_siddhi_curves(dummy_results)
