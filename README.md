# code_cascading_shared
This is the shared version of my paper, Model Cascading for Code. The code is very messy because I got rejected twice and recently submitted for the third time. I find it not worth my time to clean it up, because the idea is outdated and the model families are outdated, one of which is not supported on huggingface at the current point. I recommend you just get the idea from the most relevant scripts and go on with your own implementations. 

The idea is very simple - let the smaller models answer the question and do self-test. If the number of passing pairs is higher than total x theta (taking the lower integer), then we adopt the answer; otherwise we ask the bigger model. 

However, the experiments are not conducted in the sequence described in the above pipeline. These are the steps I did to generate the plots in my paper:
1. Select a model family (ie. WizardCoder 7B, 13B, 34B), and a dataset (ie. HumanEval).
2. Let each SINGLE MODEL answer the full dataset 10 times. Note that the greedy inference was unique so it was onlt generated once. The script is `singular.py`. Once finished, you will get folders like `code/answer/7B`. I ran 10 times on top of 10 generations, just to have a more faithful statistics of pass@1,2,3,4,5,10 accuracies. 
3. Similarly, let each SINGLE MODEL generate tests for the full dataset 10 times. I asked the models to generate as many tests as possible, so I can pick for different numbers of tests. the script is `testcase.py`. i also ran 10 times on top of 10 generations to get a more faithful statistics of having test@0,2,4.
4. To get the effect of test-passing mechanism, I created `select.py` to pairwise test the question and test and select the best pair. The output is `code/selected`.
5. Calculate stats of accuracy and cost. All the other dispersed scripts are for this step. 