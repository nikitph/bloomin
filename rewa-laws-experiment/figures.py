import matplotlib.pyplot as plt
import numpy as np
import json

# Load your data
with open('k_semantics_results.json', 'r') as f:
    data = json.load(f)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel A: OR vs AND Scaling (Top Left)
ax1 = fig.add_subplot(gs[0, 0])
K_and = data['exp1']['AND']['K']
chi_and = data['exp1']['AND']['chi_c']
K_or = data['exp1']['OR']['K']
chi_or = data['exp1']['OR']['chi_c']

ax1.plot(K_and, chi_and, 'o-', color='#2E86AB', linewidth=3, 
         markersize=10, label='AND (concatenated)')
ax1.plot(K_or, chi_or, 's-', color='#A23B72', linewidth=3, 
         markersize=10, label='OR (independent)')

# Add trend lines
z_and = np.polyfit(K_and, chi_and, 1)
z_or = np.polyfit(K_or, chi_or, 1)
x_fit = np.linspace(2, 8, 100)
ax1.plot(x_fit, np.poly1d(z_and)(x_fit), '--', color='#2E86AB', 
         alpha=0.5, label=f'AND: χ_c = {z_and[0]:.2f}K + {z_and[1]:.2f}')
ax1.plot(x_fit, np.poly1d(z_or)(x_fit), '--', color='#A23B72', 
         alpha=0.5, label=f'OR: χ_c = {z_or[0]:.2f}K + {z_or[1]:.2f}')

ax1.set_xlabel('Number of Hash Functions, K', fontsize=13)
ax1.set_ylabel('Critical Point, χ_c', fontsize=13)
ax1.set_title('A. OR vs AND: Critical Point Scaling', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'AND: χ_c ∝ K\nOR: χ_c ∝ 1/K', 
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Mutual Information Additivity Test (Top Right)
ax2 = fig.add_subplot(gs[0, 1])
K_mi = data['exp2']['K']
MI_total = data['exp2']['MI_total']
MI_per_hash = data['exp2']['MI_per_hash']

ax2.plot(K_mi, MI_total, 'o-', color='#6A4C93', linewidth=3, 
         markersize=10, label='Total MI(B; Y)')
ax2.axhline(np.mean(MI_total), color='k', linestyle='--', alpha=0.5,
            label=f'Mean = {np.mean(MI_total):.3f} ± {np.std(MI_total):.3f}')
ax2.set_xlabel('Number of Hash Functions, K', fontsize=13)
ax2.set_ylabel('Mutual Information I(B; Y)', fontsize=13)
ax2.set_title('B. MI Additivity Test: Information Does NOT Accumulate', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, 'Flat → AND regime\n(not independent channels)', 
         transform=ax2.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Panel C: C_M Finite-Size Scaling (Bottom Left)
ax3 = fig.add_subplot(gs[1, :])
K_cm = np.array(data['exp3']['K'])
L_cm = np.array(data['exp3']['L'])
C_M = np.array(data['exp3']['C_M'])

for L_val in [50, 75, 100]:
    mask = L_cm == L_val
    K_subset = K_cm[mask]
    CM_subset = C_M[mask]
    ax3.plot(K_subset, CM_subset, 'o-', linewidth=2.5, markersize=9, 
             label=f'L={L_val}')

ax3.set_xlabel('Number of Hash Functions, K', fontsize=13)
ax3.set_ylabel('Effective Constant C_M = χ_c / (K·log N)', fontsize=13)
ax3.set_title('C. Finite-Size Correction: C_M Grows with System Size', 
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=11, title='Witness Count')
ax3.grid(alpha=0.3)
ax3.text(0.05, 0.95, 'C_M increases with K and L\n→ Capacity grows with witness budget', 
         transform=ax3.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel D: Correlation Sensitivity (Bottom)
ax4 = fig.add_subplot(gs[2, :])
corr = data['exp4']['correlation']
chi_corr = data['exp4']['chi_c']

ax4.plot(corr, chi_corr, 'o-', color='#C1292E', linewidth=3, 
         markersize=12, label='Observed χ_c')
ax4.set_xlabel('Witness Correlation Level', fontsize=13)
ax4.set_ylabel('Critical Point, χ_c', fontsize=13)
ax4.set_title('D. Correlation Stress Test: Independence Assumption is Critical', 
              fontsize=14, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(alpha=0.3, which='both')
ax4.text(0.5, 0.5, 'χ_c drops 100× as correlation increases\n→ Witness independence is essential', 
         transform=ax4.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.savefig('figure_2_k_regime_investigation.png', dpi=300, bbox_inches='tight')
plt.show()