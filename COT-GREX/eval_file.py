import pandas as pd
import re
from ast import literal_eval
import ast
import numpy as np
from evaluate import load
bertscore = load("bertscore")
rouge = load('rouge')
print("rouge metric loaded")
print("bert loaded")
from tqdm import tqdm

pred_file = pd.read_csv('/scratch/work/dattap1/knowledge-graph/predictions/HyperRED/prediction_GPT3.5_gpt_1_qy_ry_qual_desc.tsv',  sep='\t', encoding='latin-1', converters={'Triplets':ast.literal_eval, 'Triplets_with_qualif': ast.literal_eval, 'gpt_pred_parsed': ast.literal_eval, 'gpt_pred_parsed_qith_qualif': ast.literal_eval, })

pred_file = pred_file.drop([])
pred_file = pred_file.dropna(subset=['RE_pred'])


def extract_triplets_from_text(string_of_tuples):
    try:
        # Use a regular expression to find all tuples in the input string
        tuple_strings = re.findall(r'\([^)]*\)', string_of_tuples)

        # Initialize an empty list to store the parsed tuples
        list_of_tuples = []

        # Iterate through each tuple string and convert to a tuple
        for tpl_str in tuple_strings:
            # Remove parentheses from each end
            tpl_str = tpl_str.strip('()')

            # Split elements within the tuple
            elements = [elem.strip() for elem in tpl_str.split(',')]

            # Convert each element to a tuple and add to the list
            parsed_tuple = tuple(elements)
            list_of_tuples.append(parsed_tuple)

        return list_of_tuples
    except Exception as e:
        print(f"Error: {e}. Unable to convert the string to a list of tuples. Original string: {string_of_tuples}")
        return None


pred_file['RE_pred'] = pred_file['RE_pred'].apply(lambda x: extract_triplets_from_text(x))

pred_file = pred_file.dropna(subset=['RE_pred'])

print(pred_file.shape)

all_true_triplets = pred_file.Triplets.values
# all_predictions = [eval(str(elem)) for elem in pred_file['RE_pred']]
all_predictions = pred_file.RE_pred.values


def find_rouge_2(ground_triplet, context_list):
    results = [0.0]

    for triplets in context_list:
        try:
            relation = [triplets[2]]
            ground_relation = [ground_triplet[2]]
            if rouge.compute(predictions=relation, references=ground_relation, use_aggregator=True)['rouge2'] >= 0.66:

                prediction_ent = [triplets[0], triplets[1]]
                ground_truth_ent = [ground_triplet[0], ground_triplet[1]]
                result = rouge.compute(predictions=prediction_ent, references= ground_truth_ent, use_aggregator=True)['rouge2']
                results.append(result)
                # if result < 0.5:
                #     print("Triplets:", triplets) 
                #     print("ground triplet", ground_triplet)

        except Exception as e:
            print(e)
            results.append(0)
    # if max(results) < 0.7:
    #     print(context_list, ground_triplet )
    if max(results) > 0.65:
        # print(results)
        return True
    return False


def find_rouge_1(ground_triplet, context_list):
    results = [0.0]
    for triplets in context_list:
        try:
            relation = [triplets[2]]
            ground_relation = [ground_triplet[2]]
            if rouge.compute(predictions=relation, references=ground_relation, use_aggregator=True)['rouge1'] >= 0.66:

                prediction_ent = [triplets[0], triplets[1]]
                ground_truth_ent = [ground_triplet[0], ground_triplet[1]]

                result = rouge.compute(predictions=prediction_ent, references= ground_truth_ent, use_aggregator=True)['rouge1']
                results.append(result)
                # if result < 0.5:
                #     print("Triplets:", triplets) 
                #     print("ground triplet", ground_triplet)

        except Exception as e:
            print(e)
            results.append(0)
    # if max(results) < 0.7:
    #     print(context_list, ground_triplet )
    if max(results) >= 0.66:
        # print(results)
        return True
    return False


def find_bert_score(ground_triplet, context_list):
#     for triplets in context_list: (if triplets[2] == ground_triplet[2] )
    results = [0.0]

    for triplets in context_list:
        try:
            relation = [triplets[2]]
            ground_relation = [ground_triplet[2]]
            if bertscore.compute(predictions=relation, references=ground_relation, lang="en", model_type="distilbert-base-uncased")['f1'][0] > 0.65:

                prediction_ent = [triplets[0], triplets[1]]
                ground_truth_ent = [ground_triplet[0], ground_triplet[1]]

                result = bertscore.compute(predictions=prediction_ent, references= ground_truth_ent, lang="en", model_type="distilbert-base-uncased")['f1'][0]
                results.append(result)
                # if result < 0.5:
                #     print("Triplets:", triplets) 
                #     print("ground triplet", ground_triplet)

        except Exception as e:
            print(e)
            results.append(0)
    # if max(results) < 0.7:
    #     print(context_list, ground_triplet )
    if max(results) > 0.65:
        # print(results)
        return True
    return False



def calculate_TP_FP_FN(true_annots: list, predicted_annots: list, mode: str, threshold=None):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

#     assert true_annots == predicted_annots.all()  # the number of contexts should be equal

    for context_gold, context_pred in tqdm(zip(true_annots, predicted_annots)):

        if mode == 'bert-score':
            for triplet_pred in context_pred:
                if find_bert_score(triplet_pred, context_gold):
                    true_positives+=1
                else:
                    false_positives+=1

            for triplet_truth in context_gold:
                if not find_bert_score(triplet_truth, context_pred):
                    false_negatives +=1

        if mode == 'rouge-1':
            for triplet_pred in context_pred:
                if find_rouge_1(triplet_pred, context_gold):
                    true_positives+=1
                else:
                    false_positives+=1

            for triplet_truth in context_gold:
                    if not find_rouge_1(triplet_truth, context_pred):
                        false_negatives +=1

        if mode == 'rouge-2':
            for triplet_pred in context_pred:
                if find_rouge_2(triplet_pred, context_gold):
                    true_positives+=1
                else:
                    false_positives+=1

            for triplet_truth in context_gold:
                if not find_rouge_2(triplet_truth, context_pred):
                    false_negatives +=1




    return true_positives, false_positives, false_negatives

TP, FP, FN = calculate_TP_FP_FN(all_true_triplets, all_predictions, mode='rouge-2')

print(TP, FP, FN)

def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

calculate_precision_recall_f1(TP, FP, FN)


TP, FP, FN = calculate_TP_FP_FN(all_true_triplets, all_predictions, mode='rouge-1')

print("TP", TP, flush=True)
print("FP", FP, flush=True)
print("FN", FN, flush=True)

def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

precision, recall, f1_score = calculate_precision_recall_f1(TP, FP, FN)
print("Precision:", precision, flush=True)
print("Recall:", recall, flush=True)
print("F1 Score:", f1_score, flush=True)
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
~                                                          