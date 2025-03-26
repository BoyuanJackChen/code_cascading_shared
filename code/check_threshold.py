import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="13B")
parser.add_argument("--threshold", type=float, default=1.0)
FLAGS = parser.parse_args()

def main(args):
    model = args.model
    loop = 0
    pass_at = 10
    testlines = 4
    threshold = args.threshold
    all_thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    selected_file = f"./selected/{model}/{model}_p{pass_at}_t{testlines}_l{loop}.json"
    all_selected_dict = json.load(open(selected_file, "r"))
    total_num = len(all_selected_dict)
    
    for threshold in all_thresholds:
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        for selected_dict in all_selected_dict:
            a = selected_dict["max_answer_num"]
            t = selected_dict["max_test_num"]
            total_product = selected_dict["total_product"]
            correct = selected_dict["indeed"]
            confidence = a*t
            prediction = (confidence >= int(total_product*threshold))
            
            if prediction and correct:
                true_positive += 1
            elif prediction and not correct:
                false_positive += 1
            elif not prediction and correct:
                false_negative += 1
            elif not prediction and not correct:
                true_negative += 1
        
        # Get true/false positive/negative rates
        true_positive_rate = round(true_positive/total_num * 100, 2)
        true_negative_rate = round(true_negative/total_num * 100, 2)
        false_positive_rate = round(false_positive/total_num * 100, 2)
        false_negative_rate = round(false_negative/total_num * 100, 2)
        print(f"WizardCoder x LLAMA {model}, pick@{pass_at}, testlines={testlines}, threshold={threshold}")
        print(f"""True positive: {true_positive_rate}%
True negative: {true_negative_rate}%
False positive: {false_positive_rate}%
False negative: {false_negative_rate}%""")
    

if __name__=="__main__":
    main(FLAGS)