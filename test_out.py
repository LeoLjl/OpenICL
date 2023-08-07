import pickle
from datasets import load_dataset
import evaluate

def evaluate_result(outputs, labels):
    metric = evaluate.load("accuracy")
    results = metric.compute(references=labels, predictions=outputs)
    return results

map_label = {"positive": 0, "negative": 1}
dataset = load_dataset('gpt3mix/sst2')

with open("icl_out.bin", "rb") as f:
    predictions = pickle.load(f)

pred = list(map(lambda x: map_label[x], predictions))
test_labels = dataset["test"]["label"]

print(evaluate_result(pred, test_labels))