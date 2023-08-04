from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForCausalLM, GPT2Model
import faiss
from typing import Dict
from transformers import pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset

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


def generate_item(template, dataset: Dict, ice_id, label_map: Dict, input_column="text", label_column="label"):
    label = dataset["train"][label_column][ice_id]
    prompt = template.format(text=dataset["train"][input_column][ice_id], verb=label_map[label])
    return prompt
    

def generate_prompts(template, dataset: Dict, ice_ids, label_map: Dict, split="test", input_column="text", label_column="label"):
    prompts = []
    
    # construct a prompt for every test data point
    for idx, neighbours in enumerate(tqdm(ice_ids)):
        prompt = "\n".join([generate_item(template, dataset, i, label_map, input_column, label_column) for i in neighbours]) + "\n"
        prompt += template.format(text=dataset[split][input_column][idx], verb="")
        prompts.append(prompt)
    
    return prompts

def data(prompts):
    for prompt in prompts:
        yield prompt

def inference1(model_name, prompts):
    pipe = pipeline("text-generation", model=model_name, device="cuda", batch_size=2, return_full_text=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    pipe.tokenizer.padding_side = "left"
    
    dataset = ListDataset(prompts)
    
    for out in tqdm(pipe(prompts)):
        s = out[0]["generated_text"]
        # import pdb; pdb.set_trace()
    

def inference(model_name, prompts):
    device = "cuda"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output = model.generate(input_ids)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Prompt: ", prompt)
        print("Generated text: ", generated_text)

    
    # tokenize prompts
    # input_ids = tokenizer(prompts, return_tensors="pt", padding=True, max_length=None).to(device)
    
    # output = model(**input_ids)


if __name__ == "__main__":
    from datasets import load_dataset

    # Loading dataset from huggingface
    dataset = load_dataset('gpt3mix/sst2')
    train_corpus = dataset["train"]["text"]
    test_corpus = dataset["test"]["text"]
    train_embeddings = get_embeddings(train_corpus)
    index = create_index(train_embeddings)
    neighbours = knn_search(index, test_corpus)
    
    template = "Review:{text}\nSentiment:{verb}"
    label_map = {0: "positive", 1: "negative"}
    prompts = generate_prompts(template, dataset, neighbours, label_map)
    print(prompts[:10])
    inference1("gpt2-xl", prompts)