"""For plot in paper
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example values for [k1, k2, k3, t1, t2, t3]
k1, k2, k3, t1, t2, t3 = 5, 3, 0, 4, 4, 0  # Replace with your values

# Initialize lists to store the data
thetas = []
accuracies = []
costs = []

# Load data from each file and extract relevant information
for theta in [i * 0.1 for i in range(11)]:  # 0.0, 0.1, 0.2, ..., 1.0
    file_name = f'../cascade_results/full_threshold{theta:.1f}.csv'
    df = pd.read_csv(file_name)
    
    # Convert k1 column to integer
    df['k1'] = df['k1'].astype(int)
    df['k2'] = df['k2'].astype(int)
    df['k3'] = df['k3'].astype(int)
    df['t1'] = df['t1'].astype(int)
    df['t2'] = df['t2'].astype(int)
    df['t3'] = df['t3'].astype(int)
    print(df)
    
    # Filter the dataframe for the specified [k1, k2, k3, t1, t2, t3] values
    row = df.loc[(df['k1'] == k1) & (df['k2'] == k2) & (df['k3'] == k3) & 
                 (df['t1'] == t1) & (df['t2'] == t2) & (df['t3'] == t3)]

    # Append the relevant values to the lists
    if not row.empty:
        thetas.append(theta)
        accuracies.append(row['accuracy'].values[0])
        costs.append(row['cost'].values[0])
        
# Recreating the plot with the specified adjustments
# Set x-axis ticks
fig, ax1 = plt.subplots(figsize=(9, 6))
plt.xticks(np.arange(min(thetas), max(thetas)+0.1, 0.1))

# Light purple color for accuracy line
accuracy_color = 'blue'
ax1.set_xlabel('Theta', color='black', fontsize=14)
ax1.set_ylabel('Accuracy (%)', color='black', fontsize=14)
ax1.plot(thetas, accuracies, color=accuracy_color, linewidth=3, label='Accuracy')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

# Darker gold color for cost line
cost_color = 'goldenrod'
ax1.spines['top'].set_visible(False)
ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)  # Also remove top line for the secondary axis

ax2.set_ylabel('Cost ($)', color='black', fontsize=14)
ax2.plot(thetas, costs, color=cost_color, linewidth=3, label='Cost per 1k queries')
ax2.tick_params(axis='y', labelcolor='black')
ax2.grid(which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgray')

fig.tight_layout(pad=3.0)  # otherwise the right y-label is slightly clipped

# Add legend on blue and yellow lines
# Creating legends from both axes and combining them
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=13)

print(accuracies)
print(costs)

# Output pdf
plt.savefig(f'../cascade_results/theta.pdf', bbox_inches='tight')
plt.show()