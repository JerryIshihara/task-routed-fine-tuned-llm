import os
import json
import torch
import re
import sqlite3
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Create the parser
parser = argparse.ArgumentParser(description="argparse")

# Add arguments
parser.add_argument("model", type=str, help="LLM Model to use")
parser.add_argument("dataset", type=str, help="The dataset to test with")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
arg_model = args.model
arg_dataset = args.dataset
MODELS = ["medicine", "finance", "SQL", "llama2", "router"]
DATASET = ["med", "fin", "nsql"]
assert arg_model in MODELS, "model not supported"
assert arg_dataset in DATASET, "dataset not supported"

logging.basicConfig(
    filename=f'log/model_{arg_model}_dataset_{arg_dataset}.log', 
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

def print_and_log(text):
    logging.info(text)
    print(text)


### ============ Load Model from Huggingface ============ ###

if arg_model in ["llama2", "router"]:
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, device_map='auto')
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
    print("LLAMA2 Model ", llama_model.device)

if arg_model in ["SQL", "router"]:
    nsql_model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-llama-2-7B", torch_dtype=torch.bfloat16, device_map='auto')
    nsql_tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-llama-2-7B")
    print("NSQL Model ", nsql_model.device)

if arg_model in ["medicine", "router"]:
    med_model = AutoModelForCausalLM.from_pretrained("AdaptLLM/medicine-chat", torch_dtype=torch.bfloat16, device_map='auto')
    med_tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/medicine-chat", use_fast=False)
    print("Med Model ", med_model.device)

if arg_model in ["finance", "router"]:
    fin_model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-chat", torch_dtype=torch.bfloat16, device_map='auto')
    fin_tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-chat", use_fast=False)
    print("Fin Model ", fin_model.device)



### ============ Load Dataset ============ ###

# Define the parent absolute path
parent_path = os.path.abspath(os.path.join(__file__, os.pardir))
SPIDER_FOLDER = os.path.join(parent_path, 'spider')
DATABASE_FOLDER = os.path.join(SPIDER_FOLDER, 'database')

# Load JSON data from a file
with open(os.path.join(SPIDER_FOLDER, "train_spider.json"), 'r') as file:
    nsql_dataset = json.load(file)

med_dataset = load_dataset("AdaptLLM/medicine-tasks", "RCT")
fin_dataset = load_dataset("AdaptLLM/finance-tasks", "Headline")

### ============ model query function ============ ###
def llama2_query(prompt, max_len=8192):
    # Tokenize the prompt
    input_ids = llama_tokenizer.encode(prompt, return_tensors="pt").to(llama_model.device)
    # Generate a response
    output = llama_model.generate(input_ids, max_length=max_len, do_sample=False)
    # Decode the response
    decoded_output = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def nsql_query(query):
    input_ids = nsql_tokenizer(query, return_tensors="pt").input_ids.to(nsql_model.device)
    generated_ids = nsql_model.generate(input_ids, max_length=4096, do_sample=False)
    return nsql_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def med_query(query):
    inputs = med_tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(med_model.device)
    outputs = med_model.generate(input_ids=inputs, max_length=8192, do_sample=False)[0]

    answer_start = int(inputs.shape[-1])
    pred = med_tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)
    return pred

def fin_query(query):
    inputs = fin_tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(fin_model.device)
    outputs = fin_model.generate(input_ids=inputs, max_length=8192, do_sample=False)[0]

    answer_start = int(inputs.shape[-1])
    pred = fin_tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)
    return pred
    

### ============ prompting templates ============ ###
def router_prompt(query):
    system_prompt = 'Classify the text into one of the categories.'
    med_example = 'Explain the role of this sentence in an abstract: "Analyses of acoustic voice parameters revealed significant lowering of average pitch in the 12.5 - and 25-mg dose groups compared to placebo (P<.05); these changes in pitch were significantly related to increases in T concentrations." Results Explain the role of this sentence in an abstract: "The SMI technique has an immediate positive effect on elbow extension in the ULNT-1." Conclusions Explain the role of this sentence in an abstract: "This study examined mediators and moderators of short-term treatment effectiveness from the iQUITT Study (Quit Using Internet and Telephone Treatment), a 3-arm randomized trial that compared an interactive smoking cessation Web site with an online social network (enhanced Internet) alone and in conjunction with proactive telephone counseling (enhanced Internet plus phone) to a static Internet comparison condition (basic Internet).'
    fin_example = 'Headline: "Gold falls to Rs 30,800; silver down at Rs 41,200 per kg" Now answer this question: Does the news headline talk about price in the past? Yes Headline: "gold futures add to gains after adp data" Now answer this question: Does the news headline talk about price? Yes Headline: "Gold holds on to modest loss after data" Now answer this question: Does the news headline talk about price in the future? No Headline: "spot gold quoted at $417.50, down 20c from new york" Now answer this question: Does the news headline talk about a general event (apart from prices) in the past? No Headline: "gold hits new record high at $1,036.20 an ounce" Now answer this question: Does the news headline compare gold with any other asset? No Headline: "gold may hit rs 31,500, but pullback rally may not sustain for long: experts" Now answer this question: Does the news headline talk about price?'
    sql_example = """PRAGMA foreign_keys = ON;
    CREATE TABLE "author" (
    "aid" int,
    "homepage" text,
    "name" text,
    "oid" int,
    primary key("aid")
    );
    CREATE TABLE "conference" (
    "cid" int,
    "homepage" text,
    "name" text,
    primary key ("cid")
    );
    CREATE TABLE "domain" (
    "did" int,
    "name" text,
    primary key ("did")
    );
    -- Using valid SQLite, answer the following questions for the tables provided above.
    -- which course has most number of registered students?
    """
    user_prompt = f"""A message can be classified as one of the following categories: medicine, finance, SQL. 
    
    ### Categories:
    <CATEGORY>SQL</CATEGORY>
    <DEFINITION>This is a text-to-SQL dataset. It involves SQL scheme and a human language question for the model to generate SQL query.</DEFINITION>
    <EXAMPLE>{sql_example}</EXAMPLE>
    
    <CATEGORY>finance</CATEGORY>
    <DEFINITION>Scraped from CNBC, the Guardian, and Reuters official websites, the headlines in these datasets reflects the overview of the U.S. economy and stock market every day for the past year to 2 years.</DEFINITION>
    <EXAMPLE>{fin_example}</EXAMPLE>
    
    <CATEGORY>medicine</CATEGORY>
    <<DEFINITION>The dataset consists of approximately 200,000 abstracts of randomized controlled trials, totaling 2.3 million sentences. Each sentence of each abstract is labeled with their role in the abstract using one of the following classes: background, objective, method, result, or conclusion.</DEFINITION>
    <EXAMPLE>{med_example}</EXAMPLE>
    
    Based on these categories, classify this message:
    <MESSAGE>{query}</MESSAGE>
    
    Please select one of the following options: SQL, medicine, finance. Keep your answer as short as one word."""
    return f"<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt}[/INST]"
     
def med_prompt(data):
    user_input = f'''
    A sentence can be classified as one of the following roles: {", ".join(data['options'])}. 
    
    {data['input']}
    
    Please only provide your choice to the last sentence. Keep your answer as short as one word.'''
    prompt = f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{user_input} [/INST]{arg_model == 'SQL' and '###ANSWER:'}"
    return prompt

def _get_nsql_schema(data):
    schema_path = os.path.join(DATABASE_FOLDER, data['db_id'], 'schema.sql')
    fd = open(schema_path, 'r')
    sql_schema = "\n".join([line.strip() for line in fd.readlines() if not line.strip().startswith("INSERT")])
    return sql_schema

def nsql_prompt(data):
    prompt = f"""{_get_nsql_schema(data)}
    -- Using valid SQLite, answer the following questions for the tables provided above.
    -- {data['question']}
    ```"""
    # system_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. Keep your answer short. You must only output the SQL query that answers the question."""
    # user_prompt = _get_nsql_user_prompt(data)
    # base_prompt = f"<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]###SQL Query:```"
    return prompt
    

def fin_prompt(data):
    user_input = f'''
    A headline can be answered in "yes" or "No". 
    
    {data['input']}
    
    Please only provide your answer to the last headline. Keep your answer as short as one word.'''
    prompt = f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{user_input} [/INST]{arg_model == 'SQL' and '###ANSWER:'}"
    return prompt


### ============ Prunning model generated text ============ ###
def split_string(s):
    # Delimiters to split the string
    delimiters = ["\\n", "/", " ", "\n", ".", '"', ",", "#"]
    # Create a regex pattern with the delimiters
    pattern = '|'.join(map(re.escape, delimiters))
    # Split the string using the compiled pattern
    return re.split(pattern, s)

def med_prunning(raw_response):
    raw_response = raw_response if "[/INST]" not in raw_response else raw_response.split("[/INST]")[-1]
    raw_response = raw_response if "###ANSWER:" not in raw_response else raw_response.split("###ANSWER:")[-1]
    res = [w if w.lower() in [l.lower() for l in ['Conclusions', 'Methods', 'Background', 'Results', 'Objective']] else "" for w in split_string(raw_response.strip())][0].strip()
    return res

def fin_prunning(raw_response):
    raw_response = raw_response if "[/INST]" not in raw_response else raw_response.split("[/INST]")[1]
    raw_response = raw_response if "###ANSWER:" not in raw_response else raw_response.split("###ANSWER:")[1]
    res = [w if w.lower() in ["yes", "no"] else "" for w in split_string(raw_response.strip())][0].strip()
    return res

def nsql_prunning(raw_response):
    raw_response = raw_response.split('```')[1].strip() if "```" in raw_response else raw_response.strip()
    answer = raw_response.split('```')[0].strip() if "```" in raw_response else raw_response
    return answer if ";" not in answer else answer.split(";")[0]

### ==================== Evaluation ============================ ###
def nsql_evaluation(response, data):
    try:
        # Connect to SQLite database
        table_name = data['db_id']
        db_path = os.path.join(DATABASE_FOLDER, table_name, f'{table_name}.sqlite')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # True labels
        cursor.execute(data['query'])
        true_rows = set(cursor.fetchall())
        # predictions
        cursor.execute(response)
        pred_rows = set(cursor.fetchall())
        print(f"Predicted: {response} \nTrue: {data['query']}")
        return 1 if pred_rows.symmetric_difference(true_rows) == set() else 0
    except Exception as e:
        print(e)
        return 0
        
def med_evaluation(response, data):
    print(f"Predicted: {response} | True: {data['options'][data['gold_index']]}")
    return 1 if response.lower() == data['options'][data['gold_index']].lower() else 0

def fin_evaluation(response, data):
    print(f"Predicted: {response} | True: {data['options'][data['gold_index']]}")
    return 1 if response.lower() == data['options'][data['gold_index']].lower() else 0
    
    

### ============================== Routing ================================= ###
def classify(context):
    prompt = router_prompt(context)
    response = llama2_query(prompt, max_len=8192)
    print(response)
    return response.split(f"[/INST]")[-1].strip().lower()

router_acc = {
    "finance": 0,
    "SQL": 0,
    "medicine": 0,
    'llama2': 0
}

scores = {
    "med": 0,
    "nsql":0,
    "fin": 0,
}

dataset = {
    "med": {
        "name": "med",
        "dataset": med_dataset,
        "num_data": 100,
        "label": "medicine",
        "golden_index": 0,
        "get_data": lambda x: med_dataset['test'][x],
        "get_prompt": lambda x: med_prompt(med_dataset['test'][x]),
        "get_classify_text": lambda x: med_dataset['test'][x]['input'][:3000],
        "evaluate": lambda res, data: med_evaluation(res, data),
        "prune": lambda res: med_prunning(res),
    },
    "nsql": {
        "name": "nsql",
        "dataset": nsql_dataset,
        "label": "SQL",
        "num_data": 200,
        "golden_index": 1,
        "get_data": lambda x: nsql_dataset[x],
        "get_prompt": lambda x: nsql_prompt(nsql_dataset[x]),
        "get_classify_text": lambda x: nsql_prompt(nsql_dataset[x]),
        "evaluate": lambda res, data: nsql_evaluation(res, data),
        "prune": lambda res: nsql_prunning(res),
    }, 
    "fin": {
        "name": "fin",
        "dataset": fin_dataset,
        "label": "finance",
        "num_data": 100,
        "golden_index": 2,
        "get_data": lambda x: fin_dataset['test'][x],
        "get_prompt": lambda x: fin_prompt(fin_dataset['test'][x]),
        "get_classify_text": lambda x: fin_dataset['test'][x]['input'][:3000],
        "evaluate": lambda res, data: fin_evaluation(res, data),
        "prune": lambda res: fin_prunning(res),
    }
}


models = {
    "llama2": {
        "query": lambda prompt: llama2_query(prompt),
    },
    "medicine": {
        "query": lambda prompt: med_query(prompt),    
    },
    "SQL": {
        "query": lambda prompt: nsql_query(prompt),
    },
    "finance": {
        "query": lambda prompt: fin_query(prompt),
    },
}



if __name__ == "__main__":
    data = dataset[arg_dataset]
    for i in range(data['num_data']):
        if arg_model == "router":
            classified = classify(data['get_classify_text'](i))
            print_and_log("Router Text: " + classified)
            if classified not in ["finance", "medicine", "SQL"]:
                if "sql" in classified.lower():
                    classified = "SQL"
                elif "finance" in classified.lower():
                    classified = "finance"
                elif "medicine" in classified.lower():
                    classified = "medicine"
                else: classified = "llama2"
        else:
            classified = arg_model
        router_acc[classified] += 1
        response = models[classified]['query'](data['get_prompt'](i))
        prunned_response = data['prune'](response)
        score = data['evaluate'](prunned_response, data['get_data'](i))
        scores[data['name']] += score
        print_and_log(f"=================== {data['name']:<7} {i:>3} ======================")
        print_and_log("Model Chosen: " + classified + " | Model Expected: " + data['name'])
        print_and_log("Predicted Label: " + prunned_response + " | True Label: " + data['label'])
        print_and_log(f"ROUTING ACC | Med: {router_acc['medicine']} - Fin: {router_acc['finance']} - SQL: {router_acc['SQL']}")
        print_and_log(f"CUM ACC | {scores[data['name']]} | {scores[data['name']] / (i + 1)}")

