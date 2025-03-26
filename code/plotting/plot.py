import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplcursors

# Load all_combinations_test.csv
seed = 16
input_file_val = f"./all_comb/{seed}_val.csv"
input_file_test = f"./all_comb/{seed}_test.csv"
model_1 = '7B'
model_2 = '13B'
model_3 = '34B'
alpha_value = 0.5
lined_alpha_value = 0.3

selected_dots = [
]
baseline_dots = [
]
for input_file in [input_file_val, input_file_test]:
    df = pd.read_csv(input_file).fillna(method='ffill')
    df[['k1', 'k2', 'k3']] = df[['k1', 'k2', 'k3']].astype(int)
    fig, ax = plt.subplots()

    # Find Pareto points
    if len(selected_dots) == 0:
        selected_df = pd.DataFrame(columns=df.columns)
        for index, row in df.iterrows():
            cost_condition = df['cost'] < row['cost']
            accuracy_condition = df['accuracy'] >= row['accuracy']
            if not any(cost_condition & accuracy_condition):
                selected_df = pd.concat([selected_df, pd.DataFrame([row.values], columns=df.columns)])
        print_array = selected_df[['k1', 'k2', 'k3']].to_numpy()
        selected_dots = print_array.tolist()
        # Print this array in python format, separated with comma
        for row in print_array:
            print(f"    [{row[0]}, {row[1]}, {row[2]}],")
    else:
        selected_rows = []
        for dot in selected_dots:
            matching_row = df[(df['k1'] == dot[0]) & (df['k2'] == dot[1]) & (df['k3'] == dot[2])]
            selected_rows.append(matching_row)
        selected_df = pd.concat(selected_rows)
        
    # Define the color for the line and the dots on the line
    colors = ['dodgerblue', 'royalblue', 'mediumblue']
    dots_color = 'lightblue'

    # Plotting primary scatter plot first
    scatter = ax.scatter(df["cost"], df["accuracy"], c=dots_color, marker='o', label='Each k1, k2, k3 combination')

    # Connecting the specific dots with a line
    combinations_1 = [[0, -1, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1], [4, -1, -1], [5, -1, -1], [10, -1, -1]]
    combinations_2 = [[-1, 0, -1], [-1, 1, -1], [-1, 2, -1], [-1, 3, -1], [-1, 4, -1], [-1, 5, -1], [-1, 10, -1]]
    combinations_3 = [[-1, -1, 0], [-1, -1, 1], [-1, -1, 2], [-1, -1, 3], [-1, -1, 4], [-1, -1, 5], [-1, -1, 10]]

    # Create an empty set with columns same as df
    subset_1 = pd.DataFrame(columns=df.columns)
    subset_2 = pd.DataFrame(columns=df.columns)
    subset_3 = pd.DataFrame(columns=df.columns)

    # Plot combinations_1
    x_vals_1, y_vals_1 = [], []
    for combo in combinations_1:
        subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2])]
        if not subset.empty:
            x_vals_1.append(subset["cost"].values[0])
            y_vals_1.append(subset["accuracy"].values[0])
            subset_1 = pd.concat([subset_1, subset])
        else:
            print(f"Combination {combo} not found in the dataframe.")
    if x_vals_1 and y_vals_1:
        ax.plot(x_vals_1, y_vals_1, c=colors[0], label=model_1, alpha=alpha_value)  # Specify color and label for combinations_1
        ax.scatter(x_vals_1, y_vals_1, c=colors[0], marker='o', alpha=lined_alpha_value)

    # Plot combinations_2
    x_vals_2, y_vals_2 = [], []
    for combo in combinations_2:
        subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2])]
        if not subset.empty:
            x_vals_2.append(subset["cost"].values[0])
            y_vals_2.append(subset["accuracy"].values[0])
            subset_2 = pd.concat([subset_2, subset])
        else:
            print(f"Combination {combo} not found in the dataframe.")
    if x_vals_2 and y_vals_2:  
        ax.plot(x_vals_2, y_vals_2, c=colors[1], label=model_2, alpha=alpha_value)  # Specify color and label for combinations_2
        ax.scatter(x_vals_2, y_vals_2, c=colors[1], marker='o', alpha=lined_alpha_value)

    # Plot combinations_3
    x_vals_3, y_vals_3 = [], []
    for combo in combinations_3:
        subset = df[(df['k1'] == combo[0]) & (df['k2'] == combo[1]) & (df['k3'] == combo[2])]
        if not subset.empty:
            x_vals_3.append(subset["cost"].values[0])
            y_vals_3.append(subset["accuracy"].values[0])
            subset_3 = pd.concat([subset_3, subset])
        else:
            print(f"Combination {combo} not found in the dataframe.")
    if x_vals_3 and y_vals_3:  
        ax.plot(x_vals_3, y_vals_3, c=colors[2], label=model_3, alpha=alpha_value)  # Specify color and label for combinations_3
        ax.scatter(x_vals_3, y_vals_3, c=colors[2], marker='o', alpha=lined_alpha_value)

    # Find and plot the pareto baseline
    if len(baseline_dots) == 0:
        subset_all = pd.concat([subset_1, subset_2, subset_3])
    else:
        baseline_rows = []
        for dot in baseline_dots:
            matching_row = df[(df['k1'] == dot[0]) & (df['k2'] == dot[1]) & (df['k3'] == dot[2])]
            baseline_rows.append(matching_row)
        subset_all = pd.concat(baseline_rows)
    pareto_all = pd.DataFrame(columns=df.columns)
    for index, row in subset_all.iterrows():
        cost_condition = subset_all['cost'] < row['cost']
        accuracy_condition = subset_all['accuracy'] >= row['accuracy']
        if not any(cost_condition & accuracy_condition):
            pareto_all = pd.concat([pareto_all, pd.DataFrame([row.values], columns=df.columns)])
    ax.scatter(pareto_all['cost'], pareto_all['accuracy'], c='purple', marker='x', label='Baseline Optimal', s=100, lw=3)
    # Print baseline best k1, k2, k3 in python array format
    print_array = pareto_all[['k1', 'k2', 'k3']].to_numpy()
    baseline_dots = print_array.tolist()
    for row in print_array:
        print(f"    [{row[0]}, {row[1]}, {row[2]}],")

    # Adding annotations to each point in the scatter plot
    labels = [f"k1={int(row['k1'])}, k2={int(row['k2'])}, k3={int(row['k3'])}" for _, row in df.iterrows()]
    ax.scatter(selected_df['cost'], selected_df['accuracy'], c='green', label='LLM-Cascading Optimal', s=80)
    # Use mplcursors for interactivity
    mplcursors.cursor(scatter, hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    if "val" in input_file:
        title = "WizardCoder-V1.0 HumanEval val set (49 questions) pick@0,1,2,3,4,5,10, averaged at 10 runs, on n x 3090"
    else:
        title = "WizardCoder-V1.0 HumanEval test set (115 questions) pick@0,1,2,3,4,5,10, averaged at 10 runs, on n x 3090"
    plt.title(title, fontsize=14)
    plt.xlabel("Total cost to run 1000 questions ($)", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()  # Adjusts the layout to fit the title and axis labels
    plt.show()
