import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# Initialize sizes for shapes in the plot
label_size = 18
title_size = 23
# legend_size = 15
darkblue_size = 180
cross_line_width = 3

# Interpolation to curves
bin_size = 40.0
num_points = 2000

# Accuracy percentage average stats
accuracy_bin_size = 1.0

# Load the datasets
threshold = 1.0
seed = 7
df1 = pd.read_csv(f'../cascade_results/{seed}/{seed}_test_threshold{threshold}.csv')
df2 = pd.read_csv(f'../cascade_results/{seed}/{seed}_pareto_threshold{threshold}.csv')

# Create a scatter plot
plt.figure(figsize=(10, 6))

lighter_green = (0.1, 0.6, 0.1)

# Scatter plot object initialization (DO NOT REMOVE)
scatter_plots = []

###############################################################################
# 1) Build the LIGHT-BLUE set: at most one of {k1,k2,k3} >= 0
###############################################################################
lightblue_data = []
for _, row in df1.iterrows():
    non_negative_count = sum(n >= 0 for n in [row['k1'], row['k2'], row['k3']])
    if non_negative_count <= 1:
        lightblue_data.append((row['cost'], row['accuracy']))

###############################################################################
# 2) Build the GREEN set: rows in df2 with 'Singular'==0, matched in df1
###############################################################################
green_data = []
for _, row in df2.iterrows():
    if row['Singular'] == 0:
        row1 = df1.loc[
            (df1['k1'] == row['k1']) &
            (df1['k2'] == row['k2']) &
            (df1['k3'] == row['k3']) &
            (df1['t1'] == row['t1']) &
            (df1['t2'] == row['t2']) &
            (df1['t3'] == row['t3']) 
        ]
        if len(row1) == 1:
            cost_val = row1['cost'].values[0]
            acc_val  = row1['accuracy'].values[0]
            green_data.append((cost_val, acc_val))

lb_df = pd.DataFrame(lightblue_data, columns=['cost','accuracy'])
gd_df = pd.DataFrame(green_data,   columns=['cost','accuracy'])

# If not enough points, skip
if len(lb_df) < 2 or len(gd_df) < 2:
    print("Not enough points to plot both curves.")
    exit()

###############################################################################
# 3) Identify the COMMON, lowest-cost point => (start_cost, start_acc)
###############################################################################
lb_points = set((x,y) for x,y in zip(lb_df['cost'], lb_df['accuracy']))
g_points  = set((x,y) for x,y in zip(gd_df['cost'], gd_df['accuracy']))
common_points = lb_points.intersection(g_points)

if len(common_points) == 0:
    print("No common point found between light-blue and green data. Will skip.")
    exit()

lowest_common = min(common_points, key=lambda t: (t[0], t[1]))
start_cost, start_acc = lowest_common

###############################################################################
# 4) Identify each dataset's maximum (cost, accuracy) => end points
###############################################################################
# Blue: max by cost first, then accuracy
highest_lb = max(lb_points, key=lambda t: (t[0], t[1]))
end_cost_blue, end_acc_blue = highest_lb

# Green: max by cost first, then accuracy
highest_gd = max(g_points, key=lambda t: (t[0], t[1]))
end_cost_green, end_acc_green = highest_gd

###############################################################################
# 5) Light-Blue: Larger Bin + Monotonic Correction + PCHIP
###############################################################################
# Sort by cost
lb_df.sort_values(by='cost', inplace=True)

# (A) Bin by cost for smoothing (increase bin_size to ~5)
bins = np.arange(lb_df['cost'].min(), lb_df['cost'].max() + bin_size, bin_size)
lb_df['cost_bin'] = pd.cut(lb_df['cost'], bins=bins)

# Group by cost_bin => average cost & accuracy in each bin
grouped_lb = lb_df.groupby('cost_bin').agg({
    'cost': 'mean',
    'accuracy': 'mean'
})
grouped_lb.dropna(inplace=True)
grouped_lb.reset_index(drop=True, inplace=True)

# (B) Force strictly non-decreasing accuracy
grouped_lb.sort_values(by='cost', inplace=True)
grouped_lb['accuracy'] = np.maximum.accumulate(grouped_lb['accuracy'])

# (C) Insert boundary points for start/end
boundary_lb = pd.DataFrame([
    {'cost': start_cost, 'accuracy': start_acc},
    {'cost': end_cost_blue, 'accuracy': end_acc_blue}
])
grouped_lb = pd.concat([grouped_lb, boundary_lb], ignore_index=True)
grouped_lb.drop_duplicates(subset=['cost'], inplace=True)
grouped_lb.sort_values(by='cost', inplace=True)

# PCHIP interpolation
x_lb = grouped_lb['cost'].values
y_lb = grouped_lb['accuracy'].values
pchip_lb = PchipInterpolator(x_lb, y_lb)

# Dense sampling for final smooth line
x_smooth_lb = np.linspace(x_lb.min(), x_lb.max(), num_points)
y_smooth_lb = pchip_lb(x_smooth_lb)

# Increase linewidth for better visibility
plt.plot(x_smooth_lb, y_smooth_lb, color='lightblue', linewidth=3,
         label='Light-Blue (constrained spline)')

###############################################################################
# 6) Green: Larger Bin + Monotonic Correction + PCHIP
###############################################################################
# Sort by cost
gd_df.sort_values(by='cost', inplace=True)

bins = np.arange(gd_df['cost'].min(), gd_df['cost'].max() + bin_size, bin_size)
gd_df['cost_bin'] = pd.cut(gd_df['cost'], bins=bins)

grouped_gd = gd_df.groupby('cost_bin').agg({
    'cost': 'mean',
    'accuracy': 'mean'
})
grouped_gd.dropna(inplace=True)
grouped_gd.reset_index(drop=True, inplace=True)

grouped_gd.sort_values(by='cost', inplace=True)
grouped_gd['accuracy'] = np.maximum.accumulate(grouped_gd['accuracy'])

boundary_gd = pd.DataFrame([
    {'cost': start_cost, 'accuracy': start_acc},
    {'cost': end_cost_green, 'accuracy': end_acc_green}
])
grouped_gd = pd.concat([grouped_gd, boundary_gd], ignore_index=True)
grouped_gd.drop_duplicates(subset=['cost'], inplace=True)
grouped_gd.sort_values(by='cost', inplace=True)

x_gd = grouped_gd['cost'].values
y_gd = grouped_gd['accuracy'].values
pchip_gd = PchipInterpolator(x_gd, y_gd)

x_smooth_gd = np.linspace(x_gd.min(), x_gd.max(), num_points)
y_smooth_gd = pchip_gd(x_smooth_gd)

# Increase linewidth for better visibility
plt.plot(x_smooth_gd, y_smooth_gd, color='green', linewidth=6,
         label='Green (constrained spline)')

###############################################################################
# 7) Finishing Touches
###############################################################################
# Make the gridlines lighter by lowering alpha
plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.2)

plt.xlabel('Cost per 1k questions ($)', fontsize=label_size)
plt.ylabel('Accuracy (%)', fontsize=label_size)
plt.title(f'WizardCoder-Python-V1.0 Family (7B, 13B, 34B)', fontsize=title_size)
plt.savefig(f"../cascade_results/wizard_avg.png", dpi=1000)
plt.show()


###############################################################################
# 8) Calculate average percentage savings
###############################################################################

# Prepare arrays for results
percentage_savings_list = []

# Loop over green points (accuracy and cost)
for green_cost, green_accuracy in zip(gd_df['cost'], gd_df['accuracy']):
    # Define bin range around the current green accuracy
    bin_lower = green_accuracy - accuracy_bin_size / 2
    bin_upper = green_accuracy + accuracy_bin_size / 2

    # Find blue points within the bin range
    blue_points_in_bin = lb_df[
        (lb_df['accuracy'] >= bin_lower) & (lb_df['accuracy'] <= bin_upper)
    ]

    # Skip if no blue points are in the bin
    if len(blue_points_in_bin) == 0:
        continue

    # Compute the average cost of blue points in the bin
    avg_blue_cost = blue_points_in_bin['cost'].mean()

    # Calculate percentage savings
    percentage_savings = 100 * (avg_blue_cost - green_cost) / avg_blue_cost
    percentage_savings_list.append(percentage_savings)

# Compute the overall average percentage savings
average_percentage_savings = np.mean(percentage_savings_list)

print(f"Average Percentage Savings: {average_percentage_savings:.2f}%")

highest_saving = max(percentage_savings_list)
second_highest_saving = sorted(percentage_savings_list)[-3]
highest_saving_index = percentage_savings_list.index(highest_saving)
second_highest_saving_index = percentage_savings_list.index(second_highest_saving)
highest_saving_green_cost = gd_df['cost'].iloc[highest_saving_index]
second_highest_saving_green_cost = gd_df['cost'].iloc[second_highest_saving_index]
highest_saving_row = df1.iloc[highest_saving_index]
second_highest_saving_row = df1.iloc[second_highest_saving_index]
print(f"Second Highest Percentage Savings: {second_highest_saving:.2f}%")
print(f"Highest Percentage Savings: {highest_saving:.2f}%")
print(f"row: {highest_saving_row}")
print(f"row: {second_highest_saving_row}")