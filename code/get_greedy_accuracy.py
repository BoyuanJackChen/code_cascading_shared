"""For the special case where a1=1, t1=0"""

import json

model = "7B"
selected_file = f"./selected/{model}/{model}_p0_t0_l0.json"

with open(selected_file, 'r') as f:
    selected_data = json.load(f)

all_indeed = 0
for selected_dict in selected_data:
    if selected_dict["indeed"]:
        all_indeed += 1

print(f"Accuracy: {all_indeed}/{len(selected_data)} = {round(all_indeed/len(selected_data)*100, 1)}%")