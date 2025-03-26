import pandas as pd
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="7B", help="Model name")
parser.add_argument("--num_loops", type=int, default=10, help="Model name")
FLAGS = parser.parse_args()

def count_total_ids(the_dict, p, t=-1):
    total_ids = 0
    current_number = -1
    current_count = 0
    for answer in the_dict:
        number = answer["number"]
        if number != current_number:
            current_number = number
            current_count = 0
        if current_count < p:
            if t<0:
                total_ids += answer["num_ids"]
            else:
                total_ids += answer[f"num_ids_{t}"]
            current_count += 1
        else:
            continue
    return total_ids

def main(args):
    # Process input parameters
    all_models = ["7B", "13B", "34B"]
    model = args.model
    all_num_loops = args.num_loops
    all_pass_at = [0,1,3,5,10]
    all_testlines = [2,4]
    df_all_costs = pd.read_csv("../../../throughput/humaneval_all_costs.csv")
    
    # Process output parameters
    output_file = f"stats_wizard_he.csv"
    df = pd.DataFrame(columns=["model", "pass_at", "testlines", "loop", "cost", "accuracy"])
    
    for model in all_models:
        per_token_cost = df_all_costs.loc[df_all_costs['Size'] == model, 'Cost per 1k tokens ($)'].iloc[0]
        for pass_at in all_pass_at:
            num_loops = all_num_loops if pass_at>1 else 1
            all_accuracy = np.zeros(num_loops)
            for loop in range(num_loops):
                # First process all answers to get the total ids for answer. This does not need variation on num testlines
                if pass_at <= 1:
                    answer_file = f"./answer/{model}/{model}_p0_l{loop}.json"
                else:
                    answer_file = f"./answer/{model}/{model}_p10_l{loop}.json"
                all_answers_dict = json.load(open(answer_file, "r"))
                total_ids_answers = count_total_ids(all_answers_dict, pass_at, t=-1)
                
                for testlines in all_testlines:
                    # Second process selected answer to get the accuracy
                    selected_file = f"./selected/{model}/{model}_p{max(pass_at,1)}_t{testlines}_l{loop}.json"
                    all_selected_dict = json.load(open(selected_file, "r"))
                    num_correct = 0
                    for selected_dict in all_selected_dict:
                        if selected_dict["indeed"]:
                            num_correct += 1
                    accuracy = num_correct / len(all_selected_dict)
                    
                    # Third process testcases and get the total ids for testcase
                    if pass_at <= 1:
                        testcase_file = f"./testcase/{model}/{model}_p0_l{loop}.json"
                    else:
                        testcase_file = f"./testcase/{model}/{model}_p10_l{loop}.json"
                    all_testcases_dict = json.load(open(testcase_file, "r"))
                    total_ids_testcases = count_total_ids(all_testcases_dict, max(pass_at,1), t=testlines)
                    
                    # Finally plus the two average costs together
                    total_ids = total_ids_answers + total_ids_testcases
                    total_cost = total_ids * per_token_cost / 1000
                    new_row_df = pd.DataFrame([{"model": model, "pass_at": pass_at, "testlines": testlines, "loop": loop, "cost": total_cost, "accuracy": accuracy}])
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    print(f"model: {model}, pass_at: {pass_at}, testlines: {testlines}, loop: {loop}, cost: {total_cost}, accuracy: {accuracy}")
            
    # Group by 'pass_at' and 'testlines', then calculate the mean of 'cost' and 'accuracy'
    df = df.groupby(['model', 'pass_at', 'testlines']).agg({
        'loop': 'last',
        'cost': 'mean', 
        'accuracy': 'mean'
    }).reset_index()
    df['loop'] += 1
    
    # Write df to file
    print(df)
    input("about to write...")
    df.to_csv(output_file, index=False)

if __name__=="__main__":
    main(FLAGS)
