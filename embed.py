from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForCausalLM, GPT2Model
import faiss
from typing import Dict
from transformers import pipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import evaluate
from datasets import load_dataset

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
        
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]
    
    
def get_embeddings(corpus, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def create_index(embeddings):
    index = faiss.IndexFlatIP(768)
    index.add(embeddings)
    return index

def knn_search(index, corpus, ice_num=8):
    print("Embedding search data...")
    query_embeddings = get_embeddings(corpus)
    
    _, neighbours = index.search(query_embeddings, ice_num)
    return neighbours    


def generate_item(template, dataset, ice_id, label_map: Dict, input_column="text", label_column="label"):
    label = dataset[label_column][ice_id]
    prompt = template.format(text=dataset[input_column][ice_id], verb=label_map[label])
    return prompt
    

def generate_prompts(template, dataset: Dict, ice_ids, label_map: Dict, split="test", input_column="text", label_column="label"):
    prompts = []
    
    # construct a prompt for every test data point
    for idx, neighbours in enumerate(tqdm(ice_ids)):
        prompt = "\n".join([generate_item(template, dataset["train"], i, label_map, input_column, label_column) for i in neighbours]) + "\n"
        prompt += template.format(text=dataset[split][input_column][idx], verb="")
        prompts.append(prompt)
    
    return prompts

def data(prompts):
    for prompt in prompts:
        yield prompt

def evaluate_result(outputs, labels):
    metric = evaluate.load("accuracy")
    results = metric.compute(references=labels, predictions=outputs)
    return results

def inference1(model_name, prompts):
    pipe = pipeline("text-generation", model=model_name, device="cuda", batch_size=8, return_full_text=False, max_length=500)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    pipe.tokenizer.padding_side = "left"
    
    dataset = ListDataset(prompts)
    
    outputs = []
    
    for out in tqdm(pipe(dataset, pad_token_id=pipe.tokenizer.eos_token_id)):
        s = out[0]["generated_text"].split("\n")[0]
        if s not in ["negative", "positive"]:
            
            import pdb; pdb.set_trace()
        outputs.append(s)
    return outputs


def inference(model_name, prompts, labels, batch_size=8):
    device = "cuda"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    labels = []
    num_batches = len(prompts) // batch_size
    for i in tqdm(range(num_batches)):
        batch = prompts[i*batch_size: (i+1)*batch_size]
        batch_labels = labels[i*batch_size: (i+1)*batch_size]
        
        tokens = tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, temperature=0.0, pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        length = input_ids.shape[1]
        output = output.detach().cpu().numpy()
        
        pred = list(output[:, length])
        label = [tokenizer.decode(p, skip_special_tokens=True) for p in pred]
        # import pdb;pdb.set_trace()
        labels.extend(label)
    return labels
        

    
    # for prompt in prompts:
    #     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    #     output = model.generate(input_ids, max_new_tokens=1)
    #     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #     prediction = generated_text.removeprefix(prompt)
    #     # print("Prompt: ", prompt)
    #     print("Prediction text: ", prediction)
    #     if prediction not in ["positive", "negative"]:
    #         import pdb; pdb.set_trace()

    
    # tokenize prompts
    # input_ids = tokenizer(prompts, return_tensors="pt", padding=True, max_length=None).to(device)
    
    # output = model(**input_ids)


if __name__ == "__main__":

    # Loading dataset from huggingface
    dataset = load_dataset('gpt3mix/sst2')
    train_corpus = dataset["train"]["text"]
    test_corpus = dataset["test"]["text"]
    train_embeddings = get_embeddings(train_corpus)
    index = create_index(train_embeddings)
    neighbours = knn_search(index, test_corpus)
    
    template = "Review:{text}\nSentiment:{verb}"
    label_map = {0: "positive", 1: "negative"}
    map_label = {"positive": 0, "negative": 1}
    prompts = generate_prompts(template, dataset, neighbours, label_map)
    print(prompts[:10])
    predictions = inference("gpt2-xl", prompts, dataset["test"]["label"])
    
    import pickle
    with open("icl_out.bin", "wb") as f:
        pickle.dump(predictions, f)
    pred = list(map(lambda x: map_label[x], predictions))
    test_labels = dataset["test"]["label"]
    print(evaluate_result(pred, test_labels))