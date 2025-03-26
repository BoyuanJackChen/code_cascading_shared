import json
import numpy as np
import pandas as pd
import multiprocessing

loop = 0
pick_at = 1
limit = 4
model_name = "7B"

answer_file = f"./selected/{model_name}/{model_name}_p{pick_at}_t{limit}_l{loop}.json"
answer_file = f"./answer/{model_name}/{model_name}_p0_l0.json"
bad_questions = []
all_correct = []

# Load HumanEval Dataset
all_questions_dict = json.load(open("../../../evaluations/humaneval/data/HumanEval_py.json", "r"))

# Load the answer file
with open(answer_file, 'r') as f:
    answer_data = json.load(f)

# Create a pandas dataframe with two columns: number and accuracy
df = pd.DataFrame(columns=["number", "accuracy"])

# [number, prompt, checkpoint, passed, answer, generated_testcode, test]
import_lines = "import math\nfrom typing import List, Tuple\n"
for i in range(len(answer_data)):
    answer_dict = answer_data[i]
    correct = False
    number = answer_dict["number"]
    answer = answer_dict["answer"]
    prompt = all_questions_dict[i]["prompt"]
    test = all_questions_dict[i]["test"]
    
    # Find the last line that starts with "def "
    def_line = ""
    lines = prompt.split("\n")
    for line in reversed(lines):
        if line.startswith("def "):
            def_line = line
            break
    def_name = def_line.split(" ")[1].split("(")[0]
    test = test[test.find("def "):]
    test = test + f"\ncheck({def_name})"
    
    full_code = import_lines + answer + "\n" + test
    
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
    
    if correct:
        all_correct.append(number)
    df.loc[len(df)] = [number, int(correct)]
    print(f"Question {number} is correct: {correct}")

accuracy = df["accuracy"].mean()
mean_accuracy = accuracy
mean_accuracy = f"{round(mean_accuracy*100, 1)}%"
print(f"\n\nMean accuracy: {mean_accuracy}, length {len(df)}")
print(all_correct)