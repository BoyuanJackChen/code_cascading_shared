import pandas as pd

all_seeds = [7]
for seed in all_seeds:
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        # Load the data
        df = pd.read_csv(f'./cascade_results/{seed}/{seed}_val_threshold{threshold}.csv')

        def is_pareto(cost, accuracy, costs, accuracies):
            """Function to determine if a point is Pareto optimal."""
            for c, a in zip(costs, accuracies):
                if c <= cost and a >= accuracy and (c < cost or a > accuracy):
                    return False
            return True

        def is_singular(row):
            """Function to determine if a row is singular."""
            non_negative_values = sum(k >= 0 for k in [row['k1'], row['k2'], row['k3']])
            return non_negative_values == 1

        # Calculate if each row is Pareto optimal
        costs = df['cost'].values
        accuracies = df['accuracy'].values
        df['Pareto'] = [is_pareto(c, a, costs, accuracies) for c, a in zip(costs, accuracies)]

        # Filter out Pareto optimal rows
        pareto_df = df[df['Pareto']].copy()
        pareto_df['Singular'] = 0  # Add singular column with default value 0

        # Filter out singular rows and check if they are Pareto optimal
        singular_df = df[df.apply(is_singular, axis=1)].copy()
        costs = singular_df['cost'].values
        accuracies = singular_df['accuracy'].values
        singular_df['Pareto'] = [is_pareto(c, a, costs, accuracies) for c, a in zip(singular_df['cost'], singular_df['accuracy'])]

        # Filter out singular Pareto rows
        singular_pareto_df = singular_df[singular_df['Pareto']].copy()
        singular_pareto_df['Singular'] = 1  # Mark these rows as singular

        # Combine normal Pareto rows and singular Pareto rows
        final_df = pd.concat([pareto_df, singular_pareto_df])

        # Drop the 'Pareto' column as it's no longer needed
        final_df = final_df.drop(columns=['Pareto'])
        
        # # Uniquefy based on k1, k2, k3, t1, t2, t3
        final_df = final_df.drop_duplicates(subset=['k1', 'k2', 'k3', 't1', 't2', 't3', 'Singular'], keep='first')

        # Save the combined DataFrame to a new CSV file
        final_df.to_csv(f'./cascade_results/{seed}/{seed}_pareto_threshold{threshold}.csv', index=False)
