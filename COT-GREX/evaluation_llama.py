{
   "cell_type": "code",
   "execution_count": 9,
   "id": "86b468b1-5cc0-45f3-b7b3-bcabf5601e2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "from ast import literal_eval\n",
    "import ast\n",
    "from sentence_transformers import SentenceTransformer, util\n",
    "from tqdm import tqdm\n",
    "from evaluate import load\n",
    "bertscore = load(\"bertscore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f86d62dc-a653-4ee4-b1c4-8d0a7da69c59",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_6899/1173014056.py:6: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.\n",
      "  pred_file = pred_file.applymap(lambda x: x.lower() if isinstance(x, str) else x)\n"
     ]
    }
   ],
   "source": [
    "# pred_file = pd.read_csv('/scratch/work/dattap1/knowledge-graph/predictions/close-domain/prediction_Llama-2-70b-chat-hf_1_qn_ry_whole_dataset.tsv',  sep='\\t', encoding='latin-1', converters={'Triplets':ast.literal_eval})\n",
    "# pred_file = pd.read_csv('/scratch/work/dattap1/knowledge-graph/predictions/close-domain/prediction_Llama-2-13b-chat-hf_1_qn_rn_whole_dataset.tsv',  sep='\\t', encoding='latin-1', converters={'Triplets':ast.literal_eval})\n",
    "# pred_file = pd.read_csv('/scratch/work/dattap1/knowledge-graph/predictions/close-domain/prediction_Llama-2-13b-chat-hf_1_qn_ry_whole_dataset.tsv',  sep='\\t', encoding='latin-1', converters={'Triplets':ast.literal_eval})\n",
    "pred_file = pd.read_csv('/scratch/work/dattap1/knowledge-graph/predictions/HyperRED/prediction_GPT3.5_gpt_1_qn_rn.tsv',  sep='\\t', encoding='latin-1', converters={'Triplets':ast.literal_eval})\n",
    "\n",
    "pred_file = pred_file.applymap(lambda x: x.lower() if isinstance(x, str) else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1a04a32c-d80b-48f0-9958-4d6cea977d44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"('islam', 'david', 'considered as')\""
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_file['RE_pred'][11]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7a6f2a0e-4b00-4231-b1a5-3282d1c1d8a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_triplets_from_text(s):\n",
    "    s = re.findall('\\(.*?\\)',s)\n",
    "    try:\n",
    "        return [eval(i) for i in s]\n",
    "    except Exception as e:\n",
    "        print(e)\n",
    "    return s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "00cbbf3d-4631-497f-a88b-9d5a2a6aeaf0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_triplets_from_text(string_of_tuples):\n",
    "    try:\n",
    "        # Use a regular expression to find all tuples in the input string\n",
    "        tuple_strings = re.findall(r'\\([^)]*\\)', string_of_tuples)\n",
    "\n",
    "        # Initialize an empty list to store the parsed tuples\n",
    "        list_of_tuples = []\n",
    "\n",
    "        # Iterate through each tuple string and convert to a tuple\n",
    "        for tpl_str in tuple_strings:\n",
    "            # Remove parentheses from each end\n",
    "            tpl_str = tpl_str.strip('()')\n",
    "\n",
    "            # Split elements within the tuple\n",
    "            elements = [elem.strip() for elem in tpl_str.split(',')]\n",
    "\n",
    "            # Convert each element to a tuple and add to the list\n",
    "            parsed_tuple = tuple(elements)\n",
    "            list_of_tuples.append(parsed_tuple)\n",
    "\n",
    "        return list_of_tuples\n",
    "    except Exception as e:\n",
    "        print(f\"Error: {e}. Unable to convert the string to a list of tuples. Original string: {string_of_tuples}\")\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "bbd2ff0f-3b11-4588-8e6b-4d1b887ec895",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n",
      "Error: expected string or bytes-like object, got 'float'. Unable to convert the string to a list of tuples. Original string: nan\n"
     ]
    }
   ],
   "source": [
    "pred_file['RE_function'] = pred_file['RE_pred'].apply(lambda x: extract_triplets_from_text(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3d38651e-89b3-4510-901a-1ec62b003d8c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"('islam', 'david', 'considered as')\""
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_file['RE_pred'][11]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "bab3b7f3-3f67-4c96-831f-f103a2fcfefc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(\"'islam'\", \"'david'\", \"'considered as'\")]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_file['RE_function'][11]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5b0ee9b-0ae1-4a3d-bba0-15dcb098052b",
   "metadata": {},
   "source": [
    "## Evaluation Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b2b86f82-6278-4cfa-a40d-e670eb61695f",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def find_bert_score(ground_triplet, context_list):\n",
    "#     for triplets in context_list: (if triplets[2] == ground_triplet[2] )\n",
    "    results = [0.0]\n",
    "    \n",
    "    for triplets in context_list:\n",
    "        try: \n",
    "            relation = [triplets[2]]\n",
    "            ground_relation = [ground_triplet[2]]\n",
    "            # print(\"relation\", relation, \"ground relation\", ground_relation)\n",
    "            if bertscore.compute(predictions=relation, references=ground_relation, lang=\"en\", model_type=\"distilbert-base-uncased\")['f1'][0] > 0.65:\n",
    "                \n",
    "                prediction_ent = [triplets[0], triplets[1]]\n",
    "                ground_truth_ent = [ground_triplet[0], ground_triplet[1]]\n",
    "                \n",
    "                result = bertscore.compute(predictions=prediction_ent, references= ground_truth_ent, lang=\"en\", model_type=\"distilbert-base-uncased\")['f1'][0]\n",
    "                results.append(result)\n",
    "                # if result < 0.5:\n",
    "                #     print(\"Triplets:\", triplets) \n",
    "                #     print(\"ground triplet\", ground_triplet)\n",
    "\n",
    "        except Exception as e:\n",
    "            # print(e)\n",
    "            results.append(0)\n",
    "    # if max(results) < 0.7:\n",
    "    #     print(context_list, ground_triplet )\n",
    "    if max(results) > 0.65: \n",
    "        # print(results)\n",
    "        return True\n",
    "    return False\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "4f1e0045-275f-4c76-9e8b-fc55656b96dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_true_triplets = pred_file.Triplets.values\n",
    "# all_predictions = [eval(str(elem)) for elem in pred_file['RE_pred']]\n",
    "all_predictions = pred_file.RE_function.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2013e0bd-9f38-4c4b-b62e-fd0649186fab",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "470it [09:00,  1.15s/it]\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "'NoneType' object is not iterable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[20], line 58\u001b[0m\n\u001b[1;32m     52\u001b[0m                     false_negatives \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m\n\u001b[1;32m     56\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m true_positives, false_positives, false_negatives\n\u001b[0;32m---> 58\u001b[0m TP, FP, FN \u001b[38;5;241m=\u001b[39m \u001b[43mcalculate_TP_FP_FN\u001b[49m\u001b[43m(\u001b[49m\u001b[43mall_true_triplets\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mall_predictions\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmode\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mbert-score\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m     60\u001b[0m \u001b[38;5;28mprint\u001b[39m(TP, FP, FN)\n",
      "Cell \u001b[0;32mIn[20], line 23\u001b[0m, in \u001b[0;36mcalculate_TP_FP_FN\u001b[0;34m(true_annots, predicted_annots, mode, threshold)\u001b[0m\n\u001b[1;32m     19\u001b[0m             false_negatives \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m\n\u001b[1;32m     22\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m mode \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mbert-score\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[0;32m---> 23\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m triplet_pred \u001b[38;5;129;01min\u001b[39;00m context_pred:\n\u001b[1;32m     24\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m find_bert_score(triplet_pred, context_gold):\n\u001b[1;32m     25\u001b[0m             true_positives\u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'NoneType' object is not iterable"
     ]
    }
   ],
   "source": [
    "\n",
    "def calculate_TP_FP_FN(true_annots: list, predicted_annots: list, mode: str, threshold=None):\n",
    "    true_positives = 0\n",
    "    false_positives = 0\n",
    "    false_negatives = 0\n",
    "\n",
    "#     assert true_annots == predicted_annots.all()  # the number of contexts should be equal\n",
    "    \n",
    "    for context_gold, context_pred in tqdm(zip(true_annots, predicted_annots)):\n",
    "        \n",
    "        if mode == 'cos-sim':\n",
    "            for triplet_pred in context_pred:\n",
    "                if find_cos_sim(triplet_pred, context_gold):\n",
    "                    true_positives+=1\n",
    "                else:\n",
    "                    false_positives+=1\n",
    "            \n",
    "            for triplet_truth in context_gold:\n",
    "                if not find_cos_sim(triplet_truth, context_pred):\n",
    "                    false_negatives +=1\n",
    "\n",
    "\n",
    "        if mode == 'bert-score':\n",
    "            for triplet_pred in context_pred:\n",
    "                if find_bert_score(triplet_pred, context_gold):\n",
    "                    true_positives+=1\n",
    "                else:\n",
    "                    false_positives+=1\n",
    "            \n",
    "            for triplet_truth in context_gold:\n",
    "                if not find_bert_score(triplet_truth, context_pred):\n",
    "                    false_negatives +=1\n",
    "\n",
    "        if mode == 'rouge-1':\n",
    "            for triplet_pred in context_pred:\n",
    "                if find_rouge_1(triplet_pred, context_gold):\n",
    "                    true_positives+=1\n",
    "                else:\n",
    "                    false_positives+=1\n",
    "            \n",
    "            for triplet_truth in context_gold:\n",
    "                    if not find_rouge_1(triplet_truth, context_pred):\n",
    "                        false_negatives +=1\n",
    "        if mode == 'rouge-2':\n",
    "            for triplet_pred in context_pred:\n",
    "                if find_rouge_2(triplet_pred, context_gold):\n",
    "                    true_positives+=1\n",
    "                else:\n",
    "                    false_positives+=1\n",
    "            \n",
    "            for triplet_truth in context_gold:\n",
    "                if not find_rouge_2(triplet_truth, context_pred):\n",
    "                    false_negatives +=1\n",
    "                        \n",
    "\n",
    "\n",
    "    return true_positives, false_positives, false_negatives\n",
    "\n",
    "TP, FP, FN = calculate_TP_FP_FN(all_true_triplets, all_predictions, mode='bert-score')\n",
    "\n",
    "print(TP, FP, FN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48ea4a38-8bad-4677-8178-308176cf2abe",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):\n",
    "    precision = true_positives / (true_positives + false_positives)\n",
    "    recall = true_positives / (true_positives + false_negatives)\n",
    "    f1_score = 2 * (precision * recall) / (precision + recall)\n",
    "    return precision, recall, f1_score\n",
    "\n",
    "calculate_precision_recall_f1(TP, FP, FN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb012432-1fed-42ad-8c88-27bdad5d7c15",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
~                                                                       