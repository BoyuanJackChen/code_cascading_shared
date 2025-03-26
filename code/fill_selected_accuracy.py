import json
import pandas as pd
import multiprocessing
import numpy as np
import os
import re

all_num_loops = 10
all_pick_at = [0,1,3,5,10]
all_testlines = [0,2,4]
model_name = "13B"

# Load HumanEval Dataset
all_questions_dict = json.load(open("../../../evaluations/humaneval/data/HumanEval_py.json", "r"))

for pick_at in all_pick_at:
    for testlines in all_testlines:
        num_loops = all_num_loops if pick_at>1 else 1
        all_accuracies = np.zeros(num_loops)
        for loop in range(num_loops):
            answer_file = f"./selected/{model_name}/{model_name}_p{pick_at}_t{testlines}_l{loop}.json"
            actual_answer_file = f"./selected/{model_name}/{model_name}_p{pick_at}_t{testlines}_l{loop}_actual.json"
            actual_answer_file = answer_file
            if not os.path.exists(answer_file):
                continue
            # Load the answer file
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            output_dict_array = []
            
            # Check if indeed is a key of answer_data[0]
            # if "indeed" in answer_data[0].keys():
            #     continue
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
                question_dict = all_questions_dict[i]
                correct = False
                number = answer_dict["number"]
                if number in df["number"].values or number<0:
                    continue
                answer = answer_dict["answer"]
                
                prompt = question_dict["prompt"]
                
                # Find the last line that starts with "def "
                test = question_dict["test"]
                test = test[test.find("def "):]
                
                full_code = import_lines + answer + "\n" + test
                # print(full_code)
                # input()
                
                def code_to_run(result_queue):
                    try:
                        exec(full_code, globals())
                        result_queue.put(True)
                    except Exception as e:
                        result_queue.put(False)

                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
                process.start()
                process.join(1)
                if process.is_alive():
                    # print("Code took too long to run!")
                    process.terminate()
                    process.join()  # Ensure termination
                    correct = False
                else:
                    correct = result_queue.get()
                process.close()
                
                df.loc[len(df)] = [number, int(correct)]
                print(f"Question {number} is correct: {correct}")
                answer_dict["indeed"] = correct
                output_dict_array.append(answer_dict)

            accuracy = df["accuracy"].mean()
            print(f"Loop {loop} accuracy: {accuracy}")
            all_accuracies[loop] = accuracy
            
            # Write output_dict_array to answer_file
            with open(actual_answer_file, 'w') as f:
                json.dump(output_dict_array, f, indent=4)
            

mean_accuracy = np.mean(all_accuracies)
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
print(f"\n\n{model_name} pick@{pick_at} mean accuracy: {mean_accuracy}")

