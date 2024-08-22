# ########### Python 3.2 #############
import requests
import argparse
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='Define experiment arguments')
parser.add_argument('--prompt_dir', type=str, default='/scratch/work/dattap1/knowledge-graph/prompts/HyperRED/hyper_1_qy_ry.txt', help='define the path to the prompt')
parser.add_argument('--data', type=str, default='HyperRED', help='what dataset are you using')
parser.add_argument('--model', type=str, default='GPT4', help='type of model we are using')
parser.add_argument('--user', type=str, default='dattap1', help='to stop very unnecessary bugs')


args = parser.parse_args()


with open(args.prompt_dir, 'r', encoding='utf-8') as file:
  PROMPT = file.read()




if args.data == 'close-domain':
  test_data = pd.read_csv('/scratch/work/dattap1/knowledge-graph/data/test_gold_collapsed.tsv', sep='\t', encoding='latin-1')
elif args.data == 'HyperRED':
  test_data = pd.read_csv('/scratch/work/dattap1/knowledge-graph/data/HyperRED_collapsed.tsv', sep='\t', encoding='utf-8')[:100]

RE_pred = []

# url = "https://aalto-openai-apigw.azure-api.net/v1/chat"
url =   "https://aalto-openai-apigw.azure-api.net/v1/chat/gpt4-8k"

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": "Insert OpenAI Key"
}
data = {
    "messages": [
        {
            "role": "system",
            "content": PROMPT
        },
        {
            "role": "user",
            "content": ""
        },
        {
            "role": "user",
            "content": "Print the entities and relations in this sentence in the format (Entity1, Entity2, Relation). You are expected to strictly adhere to this format."
        }
    ],
    "temperature": 0
}

for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
    # Access the data in each row using column names
    context = row['Context']
    data['messages'][1]['content'] = context
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        try:
            RE_pred.append(response.json()['choices'][0]['message']['content'])
        except:
           RE_pred.append("None")
    else:
        print(f"Request failed with status code {response.status_code}")
        RE_pred.append("None")
    print(index)
    time.sleep(7)

test_data[f'RE_pred'] = RE_pred

pred_name = f'/scratch/work/{args.user}/knowledge-graph/predictions/{args.data}/random_4_prediction_{args.model}_{Path(args.prompt_dir).stem}.tsv'
print(pred_name, flush=True)
test_data.to_csv(pred_name, sep='\t', encoding='utf-8')

print("CSV created", flush=True)
#     curl https://aalto-openai-apigw.azure-api.net/v1/chat \
#   -H "Content-Type: application/json" \
#   -H "Ocp-Apim-Subscription-Key: " \
#   -d '{"messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Tell me a joke"}], "temperature": 0}'


#     curl https://aalto-openai-apigw.azure-api.net/v1/chat \
#   -H "Content-Type: application/json" \
#   -H "Ocp-Apim-Subscription-Key: " \
#   -d '{"messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hi!"},{"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},{"role": "user", "content": "Hi!"}], "temperature": 0}'


~                                                                                                                                                                                                                                                                                                                                                                    
~ 