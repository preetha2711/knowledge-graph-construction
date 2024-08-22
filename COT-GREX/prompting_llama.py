from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
import argparse
from pathlib import Path
import os

# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cuda",
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model.embed_tokens": 'cuda',
    "model.layers":'cuda',
    "model.norm":'cuda'
}

tqdm.pandas()
parser = argparse.ArgumentParser(description='Define experiment arguments')
parser.add_argument('--transformer', type=str, default='meta-llama/Llama-2-13b-chat-hf', help='define the generative model to prompt')
parser.add_argument('--cache_dir', type=str, default='/scratch/work/dattap1/huggingface', help='define the directory to load the generative model')
parser.add_argument('--prompt_dir', type=str, default='/scratch/work/dattap1/knowledge-graph/prompts/1_qy_ry.txt', help='define the path to the prompt')
parser.add_argument('--data', type=str, default='HyperRED', help='what dataset are you using')
parser.add_argument('--token', type=str, default='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', help='token for llama2 model')
parser.add_argument('--temperature', type=float, default=0.8, help='set the temperature for the model'  )
parser.add_argument('--batch_size', type=int, default=1, help='set prediction batch size')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE: ', device)

with open(args.prompt_dir, 'r', encoding='utf-8') as file:
  PROMPT = file.read()

if args.data == 'close-domain':
  test_data = pd.read_csv('/scratch/work/dattap1/knowledge-graph/data/test_gold_collapsed.tsv', sep='\t', encoding='latin-1')
elif args.data == 'HyperRED':
  test_data = pd.read_csv('/scratch/work/dattap1/knowledge-graph/data/HyperRED_collapsed.tsv', sep='\t', encoding='utf-8')


print(args.transformer)
tokenizer = AutoTokenizer.from_pretrained(args.transformer)

# cache_dir=args.cache_dir,
# device_map=device_map,
# quantization_config=quantization_config,
model = AutoModelForCausalLM.from_pretrained(args.transformer,
                                              cache_dir=args.cache_dir,
                                              device_map="auto",
                                              token=args.token
                                              )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

generation_config = GenerationConfig(
    max_new_tokens=500,
    min_new_tokens=10,
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

print(torch.cuda.memory_summary(), flush=True)

def extract_from_paragraph(texts, prompt):
  if isinstance(texts, str):
    input_text = prompt.replace('{{ user_message }}', texts)
    inputs = tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(device)

  elif isinstance(texts, list) or isinstance(texts, np.ndarray):
    input_texts = [prompt.replace('{{ user_message }}', text) for text in texts]
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt", return_token_type_ids=False).to(device)

  else:
    print('Unknown data type')
  print(len(inputs), flush=True)
  outputs = model.generate(**inputs, generation_config=generation_config)

  decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

  return [decoded_pred.split('[/INST]')[1] for decoded_pred in decoded_preds]

print('MODEL LOADED, START EXTRACTING')
BATCH_SIZE = args.batch_size
RE_pred = []


for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch = test_data.iloc[i:i+BATCH_SIZE]
    print(batch['Context'].values.tolist())
    result = extract_from_paragraph(batch['Context'].values.tolist(), PROMPT)
    print(result, flush=True)
    RE_pred.extend(result)

test_data[f'RE_pred'] = RE_pred
# test_data[f'RE_pred'] = test_data.progress_apply(lambda x: extract_from_paragraph(x['Context'], PROMPT), axis=1)

# os.mkdir(f'./predictions/{args.data}')
print("creating file")

pred_name = f'/scratch/work/dattap1/prompting/knowledge-graph/predictions/{args.data}/prediction_{Path(args.transformer).stem}_{Path(args.prompt_dir).stem}_HyperRED.tsv'

test_data.to_csv(pred_name, sep='\t', encoding='utf-8')

