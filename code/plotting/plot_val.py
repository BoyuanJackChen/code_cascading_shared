import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

threshold = 1.0
seed = 7
# Load the datasets
df1 = pd.read_csv(f'./cascade_results/{seed}/{seed}_val_threshold{threshold}.csv')
df2 = pd.read_csv(f'./cascade_results/{seed}/{seed}_pareto_threshold{threshold}.csv')

# Create a scatter plot for the first dataset
plt.figure(figsize=(10, 6))

# First plot all light blue dots
for _, row in df1.iterrows():
    non_negative_count = sum(n >= 0 for n in [row['k1'], row['k2'], row['k3']])
    if non_negative_count > 1:
        plt.scatter(row['cost'], row['accuracy'], color='lightblue', zorder=1)

# Then plot all the slightly darker blue dots
for _, row in df1.iterrows():
    non_negative_count = sum(n >= 0 for n in [row['k1'], row['k2'], row['k3']])
    if non_negative_count <= 1:
        plt.scatter(row['cost'], row['accuracy'], color='mediumblue', zorder=2)  # Slightly lighter than dark blue

# Draw grid lines
plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

# Adding hover labels
annot = plt.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = f"k1-k3, t1-t3 values: {df1.iloc[ind['ind'][0]][['k1', 'k2', 'k3', 't1', 't2', 't3']].tolist()}"
    annot.set_text(text)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == scatter.axes:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            plt.draw()
        else:
            if vis:
                annot.set_visible(False)
                plt.draw()

plt.connect("motion_notify_event", hover)

# Overlay the second dataset (green dots and purple crosses)
for _, row in df2.iterrows():
    color = 'green' if row['Singular'] == 0 else 'purple'
    marker = 'o' if row['Singular'] == 0 else 'x'
    size = 40 if color == 'purple' else 20  # Increase size for purple crosses
    plt.scatter(row['cost'], row['accuracy'], color=color, marker=marker, s=size, zorder=3)

# Add legends
plt.scatter([], [], color='lightblue', label='Light Blue Dots: Non-Singular')
plt.scatter([], [], color='mediumblue', label='Medium Blue Dots: Singular')
plt.scatter([], [], color='green', label='Green Dots')
plt.scatter([], [], color='purple', marker='x', s=40, label='Purple Crosses: Larger and Thicker')

plt.xlabel('Cost ($)', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.title(f'WizardCoder-Python-V1.0 on HumanEval, pick@0,1,3,5,10, testlines=2,4, threshold={threshold}, train set', fontsize=14)

# Shift the legend to lower right
plt.legend(loc='lower right')

plt.show()
