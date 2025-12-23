
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data manually transcribed from interrupted benchmark logs
data = [
    {'N': 50000, 'classical_time': 1.3562, 'obds_query': 0.1782},
    {'N': 100000, 'classical_time': 2.7685, 'obds_query': 0.1419},
    {'N': 150000, 'classical_time': 4.1848, 'obds_query': 0.1439},
    {'N': 200000, 'classical_time': 5.7751, 'obds_query': 0.1468},
    {'N': 250000, 'classical_time': 7.0567, 'obds_query': 0.1476},
    {'N': 300000, 'classical_time': 8.5289, 'obds_query': 0.1473},
    {'N': 350000, 'classical_time': 10.0833, 'obds_query': 0.1544},
    {'N': 400000, 'classical_time': 11.5312, 'obds_query': 0.1423},
    {'N': 450000, 'classical_time': 12.9454, 'obds_query': 0.1477},
    {'N': 500000, 'classical_time': 14.4499, 'obds_query': 0.1398},
    {'N': 550000, 'classical_time': 15.6398, 'obds_query': 0.1402},
    {'N': 600000, 'classical_time': 17.2565, 'obds_query': 0.1499},
    {'N': 650000, 'classical_time': 18.9336, 'obds_query': 0.1434},
    {'N': 700000, 'classical_time': 20.5252, 'obds_query': 0.1559},
    {'N': 750000, 'classical_time': 21.8890, 'obds_query': 0.1504},
    {'N': 800000, 'classical_time': 23.3716, 'obds_query': 0.1450},
    {'N': 850000, 'classical_time': 24.5143, 'obds_query': 0.1500},
    {'N': 900000, 'classical_time': 26.0565, 'obds_query': 0.1458},
    {'N': 950000, 'classical_time': 27.9619, 'obds_query': 0.1357},
    {'N': 1000000, 'classical_time': 28.7383, 'obds_query': 0.1570},
    {'N': 1050000, 'classical_time': 32.5177, 'obds_query': 0.1301},
    {'N': 1100000, 'classical_time': 32.3459, 'obds_query': 0.2240},
    {'N': 1150000, 'classical_time': 33.9838, 'obds_query': 0.1472},
    {'N': 1200000, 'classical_time': 35.6198, 'obds_query': 0.1743},
    {'N': 1250000, 'classical_time': 37.6101, 'obds_query': 0.1484},
    {'N': 1300000, 'classical_time': 38.8288, 'obds_query': 0.1481},
    {'N': 1350000, 'classical_time': 40.7299, 'obds_query': 0.1512},
    {'N': 1400000, 'classical_time': 41.8756, 'obds_query': 0.1762},
    {'N': 1450000, 'classical_time': 47.0886, 'obds_query': 0.1635},
    {'N': 1500000, 'classical_time': 49.8968, 'obds_query': 0.1590},
    {'N': 1550000, 'classical_time': 46.9880, 'obds_query': 0.1562},
    {'N': 1600000, 'classical_time': 48.8855, 'obds_query': 0.1752},
    {'N': 1650000, 'classical_time': 51.3352, 'obds_query': 0.1492},
    {'N': 1700000, 'classical_time': 50.8726, 'obds_query': 0.1508},
    {'N': 1750000, 'classical_time': 54.1956, 'obds_query': 0.2874},
    {'N': 1800000, 'classical_time': 59.0961, 'obds_query': 0.1489},
    {'N': 1850000, 'classical_time': 57.1062, 'obds_query': 0.1471},
    {'N': 1900000, 'classical_time': 57.0921, 'obds_query': 0.1465},
    {'N': 1950000, 'classical_time': 58.2617, 'obds_query': 0.1559},
    {'N': 2000000, 'classical_time': 59.5522, 'obds_query': 0.1416}
]

df = pd.DataFrame(data)

# Calculate speedup
df['speedup'] = df['classical_time'] / df['obds_query']

# Plot
plt.figure(figsize=(12, 8))
plt.loglog(df['N'], df['classical_time'], 'o-', label='Classical (O(N) Scipy)', linewidth=2)
plt.loglog(df['N'], df['obds_query'], 's-', label='OBDS Query (O(1) MPS)', linewidth=3, color='red')

plt.xlabel('Number of Points (N)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('High-Scale (MPS): Classical vs OBDS Query (N=50k to 2M)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Annotation for max speedup
max_row = df.loc[df['N'].idxmax()]
plt.annotate(f"{max_row['speedup']:.1f}x Speedup", 
             xy=(max_row['N'], max_row['obds_query']), 
             xytext=(max_row['N'], max_row['obds_query']*5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plot_path = 'obds_void_experiment/mps_partial_2m.png'
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")

# Print latex table snippet
print("\nLaTeX Table Snippet:")
print("\\begin{tabular}{|r|r|r|r|}")
print("\\hline")
print("N & Classical (s) & OBDS Query (s) & Speedup \\\\")
print("\\hline")
for _, row in df.iloc[::5].iterrows(): # Sample every 5th row
    print(f"{int(row['N']):,} & {row['classical_time']:.4f} & {row['obds_query']:.4f} & {row['speedup']:.1f}x \\\\")
print("\\hline")
print("\\end{tabular}")
