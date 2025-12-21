import matplotlib.pyplot as plt
import numpy as np

tasks = ['MNIST', 'CIFAR-10', 'SVHN', 'USPS']
baseline_forgetting = [42.45, 48.08, 17.62, 0.0]
nirodha_forgetting = [0.01, 0.05, 0.02, 0.0]

baseline_trajectory = [98.86, 66.65, 51.07, 60.43]
nirodha_trajectory = [98.81, 65.84, 62.11, 69.31]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Forgetting comparison
x = np.arange(len(tasks))
width = 0.35

ax1.bar(x - width/2, baseline_forgetting, width, 
        label='Baseline', color='#e74c3c', alpha=0.8)
ax1.bar(x + width/2, nirodha_forgetting, width,
        label='Nirodha-D (β=10000)', color='#2ecc71', alpha=0.8)

ax1.set_ylabel('Catastrophic Forgetting (%)', fontsize=12)
ax1.set_title('Forgetting Across Diverse Domains', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

# Plot 2: Average retention over task sequence
ax2.plot(range(1, 5), baseline_trajectory, 
         'o-', linewidth=2, markersize=8,
         label='Baseline', color='#e74c3c')
ax2.plot(range(1, 5), nirodha_trajectory,
         's-', linewidth=2, markersize=8, 
         label='Nirodha-D', color='#2ecc71')

ax2.set_xlabel('Task Sequence', fontsize=12)
ax2.set_ylabel('Avg Accuracy on Prev Tasks (%)', fontsize=12)
ax2.set_title('Knowledge Retention Over Sequence', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(tasks, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('dataset_rotation_complete.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graph saved as dataset_rotation_complete.png")
