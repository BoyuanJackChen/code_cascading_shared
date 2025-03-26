import json
import pandas as pd
import multiprocessing
import numpy as np
import random
import os
from itertools import combinations, combinations_with_replacement, product, permutations

model = "full"
data_folder = "./selected"
model_1 = "7B"
model_2 = "13B"
model_3 = "34B"
all_pick_at = [-1,0,1,3,5,10]
all_testlines = [0,2,4]
all_thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
all_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_loops = 10
all_numbers = list(range(0,164))
all_seeds = [7]

# Load cost per 1k tokens
df_all_costs = pd.read_csv("../../../throughput/humaneval_all_costs.csv")
cpt_1 = df_all_costs.loc[df_all_costs['Size']==model_1, 'Cost per 1k tokens ($)'].iloc[0]
cpt_2 = df_all_costs.loc[df_all_costs['Size']==model_2, 'Cost per 1k tokens ($)'].iloc[0]
cpt_3 = df_all_costs.loc[df_all_costs['Size']==model_3, 'Cost per 1k tokens ($)'].iloc[0]
print(cpt_1, cpt_2, cpt_3)

# Initialize all combinations of k and testlines
combs = list(product(all_pick_at, repeat=3))
def is_valid_combination(comb):
    # Exclude combinations where all entries are -1
    if set(comb) == {-1}:
        return False
    # Exclude combinations where an early element is 0, and a later element has a non-negative value.
    after_zero = False
    for val in comb:
        if after_zero and val >= 0:
            return False
        if val == 0:
            after_zero = True
    # Exclude combinations where there is a 1 followed by all -1's, or 1 is at the last index and all else is -1
    if 1 in comb:
        index_of_1 = comb.index(1)
        if all(x == -1 for x in comb[index_of_1+1:]) or (index_of_1 == len(comb) - 1 and all(x == -1 for x in comb[:index_of_1])):
            return False
        if comb[-1] == 1:
            return False
    return True
all_k_combos = [perm for comb in combs for perm in set(permutations(comb)) if is_valid_combination(perm)]
all_k_combos = list(set(all_k_combos))
all_k_combos = sorted(all_k_combos, key=lambda x: (x[0], x[1], x[2]))
all_t_combos = list(product(all_testlines, repeat=3))

def is_bad_combo(k, t, l):
    if k<=0 and t>0:
        return True
    if k>0 and t==0:
        return True
    return False

# mkdir if cascade_results folder does not exist
if not os.path.exists("./cascade_results"):
    os.mkdir("./cascade_results")

for seed in all_seeds:
    random.seed(seed)
    selected_numbers = random.sample(range(0, 164), 49)
    val_numbers = [num for num in selected_numbers]
    test_numbers = [num for num in all_numbers if num not in selected_numbers]
    if not os.path.exists(f"./cascade_results/{seed}"):
        os.mkdir(f"./cascade_results/{seed}")
    
    for threshold in all_thresholds:
        # selected_numbers = unselected_numbers
        output_file_name_val = f"./cascade_results/{seed}/{seed}_val_threshold{threshold}.csv"
        output_file_name_test = f"./cascade_results/{seed}/{seed}_test_threshold{threshold}.csv"

        df_result = pd.DataFrame(columns=["k1", "k2", "k3", "t1", "t2", "t3", "loop", "cost", "accuracy"])

        # for selected_numbers, output_file_name in zip([val_numbers, test_numbers], [output_file_name_val, output_file_name_test]):
        if model == "val":
            output_file_name = output_file_name_val
            selected_numbers = val_numbers
        elif model == "test":
            output_file_name = output_file_name_test
            selected_numbers = test_numbers
        else:
            output_file_name = f"./cascade_results/{seed}/full_threshold{threshold}.csv"
            selected_numbers = all_numbers
        # print(selected_numbers)
        # print(len(selected_numbers))
        # input()
        for (k1, k2, k3) in all_k_combos:
            for (t1, t2, t3) in all_t_combos:
                this_num_loops = 1 if (k1<=1 and k2<=1 and k3<=1) else num_loops
                for loop in range(this_num_loops):
                    if is_bad_combo(k1, t1, loop) or is_bad_combo(k2, t2, loop) or is_bad_combo(k3, t3, loop):
                        continue
                    loop1 = loop if k1>1 else 0
                    loop2 = loop if k2>1 else 0
                    loop3 = loop if k3>1 else 0
                    
                    total_cost = 0.0
                    total_correct = 0
                    all_numbers_left = selected_numbers + []
                    
                    if k1>=0:
                        selected_file_1 = f"./selected/{model_1}/{model_1}_p{k1}_t{t1}_l{loop1}.json"
                        selected_answers_1 = json.load(open(selected_file_1, "r"))
                        for selected_dict in selected_answers_1:
                            number = selected_dict["number"]
                            if number in all_numbers_left:
                                total_cost += selected_dict["num_ids"] * cpt_1
                                a = selected_dict["max_answer_num"]
                                t = selected_dict["max_test_num"]
                                total_product = selected_dict["total_product"]
                                confidence = a*t
                                adopt = (confidence >= total_product*threshold)
                                if k2<0 and k3<0:
                                    adopt = True
                                if adopt:
                                    all_numbers_left.remove(number)
                                    if selected_dict["indeed"]:
                                        total_correct += 1
                    
                    if k2>=0:
                        selected_file_2 = f"./selected/{model_2}/{model_2}_p{k2}_t{t2}_l{loop2}.json"
                        selected_answers_2 = json.load(open(selected_file_2, "r"))
                        for selected_dict in selected_answers_2:
                            number = selected_dict["number"]
                            if number in all_numbers_left:
                                total_cost += selected_dict["num_ids"] * cpt_2
                                a = selected_dict["max_answer_num"]
                                t = selected_dict["max_test_num"]
                                total_product = selected_dict["total_product"]
                                confidence = a*t
                                adopt = (confidence >= total_product*threshold)
                                if k3<0:
                                    adopt = True
                                if adopt:
                                    all_numbers_left.remove(number)
                                    if selected_dict["indeed"]:
                                        total_correct += 1
                    
                    if k3>=0:
                        selected_file_3 = f"./selected/{model_3}/{model_3}_p{k3}_t{t3}_l{loop3}.json"
                        selected_answers_3 = json.load(open(selected_file_3, "r"))
                        for selected_dict in selected_answers_3:
                            number = selected_dict["number"]
                            if number in all_numbers_left:
                                total_cost += selected_dict["num_ids"] * cpt_3
                                a = selected_dict["max_answer_num"]
                                t = selected_dict["max_test_num"]
                                total_product = selected_dict["total_product"]
                                all_numbers_left.remove(number)
                                if selected_dict["indeed"]:
                                    total_correct += 1
                                    
                    total_accuracy = total_correct / len(selected_numbers)
                    df_result.loc[len(df_result)] = [k1, k2, k3, t1, t2, t3, loop, total_cost, total_accuracy]
                    print(f"k1: {k1}, k2: {k2}, k3: {k3}, t1: {t1}, t2: {t2}, t3: {t3}, loop: {loop}, cost: {total_cost}, accuracy: {total_accuracy}")

        # Write df_result
        avg_df = df_result.groupby(['k1', 'k2', 'k3', 't1', 't2', 't3']).agg({
            'loop': 'last',
            'cost':'mean', 
            'accuracy':'mean'
        }).reset_index()
        # Convert ks and ts to integers
        avg_df['k1'] = avg_df['k1'].astype(int)
        avg_df['k2'] = avg_df['k2'].astype(int)
        avg_df['k3'] = avg_df['k3'].astype(int)
        avg_df['t1'] = avg_df['t1'].astype(int)
        avg_df['t2'] = avg_df['t2'].astype(int)
        avg_df['t3'] = avg_df['t3'].astype(int)
        avg_df['loop'] = avg_df['loop'].astype(int)
        avg_df['accuracy'] = avg_df['accuracy'] * 100
        # Divide cost by 1000, because cpt is in 1000 tokens; also divide by the number of questions
        avg_df['cost'] = avg_df['cost']*1000/(1000*len(selected_numbers))
        
        avg_df.to_csv(output_file_name, index=False)