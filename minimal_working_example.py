#!/usr/bin/env python3

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Set up the LM
# turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
# lm = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2')
lm = dspy.HFClientVLLM(model="TheBloke/Xwin-LM-70B-V0.1-AWQ", port=8427, url="http://localhost")

# lm = dspy.HFModel(model = 'mistralai/Mistral-7B-v0.1')
# lm = dspy.HFModel(model = 'mistral-community/Mixtral-8x22B-v0.1-AWQ')
# lm = dspy.HFModel(model = 'mistral-community/TheBloke/Mixtral_7Bx2_MoE-GPTQ')

# dspy.settings.configure(lm=turbo)
dspy.settings.configure(lm=lm)

# Load math questions from the GSM8K dataset
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]
# gsm8k_trainset, gsm8k_devset = gsm8k.train[:100], gsm8k.dev[:100]

for i in range(10):
    print(gsm8k_trainset[i])

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)


from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
# evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=5)

# Evaluate our `optimized_cot` program.
evaluation = evaluate(optimized_cot)
print("evaluation:", evaluation)

inspect_output = lm.inspect_history(n=1)
print("inspect_output:", inspect_output)





