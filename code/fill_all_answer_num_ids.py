import json
import pandas as pd
import multiprocessing
import numpy as np
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList

all_num_loops = 10
all_pick_at = [0,10]
model_name = "7B"
folder = "answer"

# Load Tokenizer
checkpoint = f"WizardLM/WizardCoder-Python-{model_name}-V1.0"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

for pick_at in all_pick_at:
    num_loops = all_num_loops if pick_at>1 else 1
    all_accuracies = np.zeros(num_loops)
    for loop in range(num_loops):
        answer_file = f"./{folder}/{model_name}/{model_name}_p{pick_at}_l{loop}.json"
        actual_answer_file = f"./{folder}/{model_name}/{model_name}_p{pick_at}_l{loop}_actual.json"
        actual_answer_file = answer_file
        if not os.path.exists(answer_file):
            continue
        # Load the answer file
        with open(answer_file, 'r') as f:
            answer_data = json.load(f)
        output_dict_array = []
        
        print(f"Working on {model_name}, {pick_at}, {loop}")

        # Create a pandas dataframe with two columns: number and accuracy
        df = pd.DataFrame(columns=["number", "accuracy"])

        multiple_pass = False
        all_keys = answer_data[0].keys()
        if "pass" in all_keys:
            multiple_pass = True

        import_lines = "import math\nfrom typing import List\n"
        for i in range(len(answer_data)):
            answer_dict = answer_data[i]
            number = answer_dict["number"]
            if number in df["number"].values or number<0:
                continue
            answer = answer_dict["answer"]
            answer_ids = tokenizer.encode(answer, return_tensors="pt")
            answer_ids_length = answer_ids.size(1) + 1
            # print(answer)
            # print(answer_ids_length)
            # input()
            
            answer_dict["num_ids"] = answer_ids_length
            output_dict_array.append(answer_dict)
        
        # Write output_dict_array to answer_file
        with open(actual_answer_file, 'w') as f:
            json.dump(output_dict_array, f, indent=4)
            

