#!/usr/bin/env python3

import re
import os
import sys

# repo_clone_path = os.path.join(os.getcwd(), 'DSPy_TweetGen_Cache')
# os.environ["DSP_NOTEBOOK_CACHEDIR"] = repo_clone_path

import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import deduplicate
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
#turbo = dspy.OpenAI(model='gpt-3.5-turbo-0613', max_tokens=500)
turbo = dspy.HFClientVLLM(model="TechxGenus/Meta-Llama-3-8B-GPTQ", port=38242, url="http://localhost")

dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)

dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
trainset = [x.with_inputs('question', 'answer') for x in dataset.train]
devset = [x.with_inputs('question', 'answer') for x in dataset.dev]

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateTweet(dspy.Signature):
    """Generate an engaging tweet that effectively answers a question staying faithful to the context, is less than 280 characters, and has no hashtags."""
    question = dspy.InputField()
    context = dspy.InputField(desc="may contain relevant facts")
    tweet = dspy.OutputField()

class Tweeter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_tweet = dspy.ChainOfThought(GenerateTweet)

    def forward(self, question, answer):
        context = []
        max_hops=2
        passages_per_hop=3
        generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        retrieve = dspy.Retrieve(k=passages_per_hop)
        for hop in range(max_hops):
            query = generate_query[hop](context=context, question=question).query
            passages = retrieve(query).passages
            context = deduplicate(context + passages)
        generated_tweet = self.generate_tweet(question=question, context=context).tweet
        return dspy.Prediction(generated_tweet=generated_tweet, context=context)
    
tweeter = Tweeter()

def has_no_hashtags(text):
    return len(re.findall(r"#\w+", text)) == 0

def is_within_length_limit(text, length_limit=280):
    return len(text) <= length_limit

def is_assessment_yes(assessment_answer):
    """Check if the first word of the assessment answer is 'yes'."""
    return assessment_answer.split()[0].lower() == 'yes'

def has_correct_answer(text, answer):
    return answer in text


class AssessTweet(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc='ignore if N/A')
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

def no_hashtags_metric(gold, pred, trace=None):
    tweet = pred.generated_tweet
    no_hashtags = has_no_hashtags(tweet)
    score = no_hashtags
    return score

def is_correct_metric(gold, pred, trace=None):
    answer, tweet = gold.answer, pred.generated_tweet
    correct = has_correct_answer(tweet, answer)
    score = correct
    return score

def within_length_metric(gold, pred, trace=None):
    tweet = pred.generated_tweet
    within_length_limit = is_within_length_limit(tweet, 280)
    score = within_length_limit
    return score

def engaging_metric(gold, pred, trace=None):
    tweet = pred.generated_tweet
    engaging = "Does the assessed text make for a self-contained, engaging tweet? Say no if it is not engaging."
    engaging = dspy.Predict(AssessTweet)(context='N/A', assessed_text=tweet, assessment_question=engaging)
    engaging = engaging.assessment_answer.split()[0].lower() == 'yes'
    score = engaging
    return score

def faithful_metric(gold, pred, trace=None):
    context, tweet = pred.context, pred.generated_tweet
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."   
    faithful = dspy.Predict(AssessTweet)(context=context, assessed_text=tweet, assessment_question=faithful)
    faithful = faithful.assessment_answer.split()[0].lower() == 'yes'
    score = faithful
    return score

def overall_metric(gold, pred, trace=None):
    answer, context, tweet = gold.answer, pred.context, pred.generated_tweet
    no_hashtags = has_no_hashtags(tweet)
    within_length_limit = is_within_length_limit(tweet, 280)
    correct = has_correct_answer(tweet, answer)
    engaging = "Does the assessed text make for a self-contained, engaging tweet? Say no if it is not engaging."
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."   
    faithful = dspy.Predict(AssessTweet)(context=context, assessed_text=tweet, assessment_question=faithful)
    engaging = dspy.Predict(AssessTweet)(context='N/A', assessed_text=tweet, assessment_question=engaging)
    engaging, faithful = [m.assessment_answer.split()[0].lower() == 'yes' for m in [engaging, faithful]]
    score = (correct + engaging + faithful + no_hashtags + within_length_limit) if correct and within_length_limit else 0
    return score / 5.0

if False:
    metrics = [no_hashtags_metric, is_correct_metric, within_length_metric, engaging_metric, faithful_metric, overall_metric]

    for metric in metrics:
        evaluate = Evaluate(metric=metric, devset=devset, num_threads=16, display_progress=True, display_table=5)
        evaluate(tweeter)

    example = devset[118]
    # print("example:", example)
    # print("example.question:", example.question)
    # print("example.answer:", example.answer)
    tweet = tweeter(question=example.question, answer = example.answer)
    print(f'Generated Tweet: ', tweet.generated_tweet)
    tweet.context

class TweeterWithAssertions(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_tweet = dspy.ChainOfThought(GenerateTweet)

    def forward(self, question, answer):
        context = []
        max_hops=2
        passages_per_hop=3
        generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        retrieve = dspy.Retrieve(k=passages_per_hop)
        for hop in range(max_hops):
            query = generate_query[hop](context=context, question=question).query
            passages = retrieve(query).passages
            context = deduplicate(context + passages)
        generated_tweet = self.generate_tweet(question=question, context=context).tweet
        dspy.Suggest(has_no_hashtags(generated_tweet), f"Please revise the tweet to remove hashtag phrases following it.", target_module=GenerateTweet)
        dspy.Suggest(is_within_length_limit(generated_tweet, 280), f"Please ensure the tweet is within {280} characters.", target_module=GenerateTweet)
        dspy.Suggest(has_correct_answer(generated_tweet, answer), "The tweet does not include the correct answer to the question. Please revise accordingly.", target_module=GenerateTweet)
        engaging_question = "Does the assessed text make for a self-contained, engaging tweet? Say no if it is not engaging."
        engaging_assessment = dspy.Predict(AssessTweet)(context=context, assessed_text=generated_tweet, assessment_question=engaging_question)
        dspy.Suggest(is_assessment_yes(engaging_assessment.assessment_answer), "The text is not engaging enough. Please revise to make it more captivating.", target_module=GenerateTweet)
        faithful_question = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
        faithful_assessment = dspy.Predict(AssessTweet)(context='N/A', assessed_text=generated_tweet, assessment_question=faithful_question)
        dspy.Suggest(is_assessment_yes(faithful_assessment.assessment_answer), "The text contains unfaithful elements or significant facts not in the context. Please revise for accuracy.", target_module=GenerateTweet)
        return dspy.Prediction(generated_tweet=generated_tweet, context=context)

tweeter_with_assertions = assert_transform_module(TweeterWithAssertions().map_named_predictors(Retry), backtrack_handler) 

if False:
    metrics = [no_hashtags_metric, is_correct_metric, within_length_metric, engaging_metric, faithful_metric, overall_metric]

    for metric in metrics:
        evaluate = Evaluate(metric=metric, devset=devset, num_threads=16, display_progress=True, display_table=5)
        evaluate(tweeter_with_assertions)

if False:
    teleprompter = BootstrapFewShotWithRandomSearch(metric = overall_metric, max_bootstrapped_demos=2, num_candidate_programs=6, num_threads=32)
    compiled_with_assertions_tweeter = teleprompter.compile(student=tweeter, teacher = tweeter_with_assertions, trainset=trainset, valset=devset[:100])
    print('Compiled with assertions Tweeter:', compiled_with_assertions_tweeter)


    for metric in metrics:
        evaluate = Evaluate(metric=metric, devset=devset, num_threads=32, display_progress=True, display_table=5)
        evaluate(compiled_with_assertions_tweeter)

# teleprompter = BootstrapFewShotWithRandomSearch(metric = overall_metric, max_bootstrapped_demos=2, num_candidate_programs=6, num_threads=32)
teleprompter = BootstrapFewShotWithRandomSearch(metric = overall_metric, max_bootstrapped_demos=8, num_candidate_programs=24, num_threads=32)
compiled_tweeter_with_assertions = teleprompter.compile(student=tweeter_with_assertions, teacher = tweeter_with_assertions, trainset=trainset, valset=devset[:100])

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=32, display_progress=True, display_table=5)
    evaluate(compiled_tweeter_with_assertions)
