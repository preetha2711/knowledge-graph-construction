import csv
import pandas as pd
from prediction import run_predict
import json
from data_process import Sentence, Entity, Relation, Data
from tqdm import tqdm
import time

omnia = pd.read_csv('./data_processed.csv')
omnia = omnia[0:4000]
data_cube_RE = pd.DataFrame({'url': ['test'],
                             'input_text': ['test'],
                             'Entity1': ['test'],
                             'Relationship': ['test'],
                             'Entity2': ['test'],
                             'Qualifiers': ['test']})


for i, row in tqdm(omnia.iterrows()):
    try:
        url = row['url']
        text = row['text']
        texts = [text]
        split_text = text.split()
        print(texts, flush=True)
        preds = run_predict(texts, path_checkpoint="./cube_model")
        print(preds, flush=True)
        preds.save("tmp.json")
        time.sleep(2)

    #     preds = sents=[Sentence(tokens=['Leonard', 'Parker', 'received', 'his', 'PhD', 'from', 'Harvard', 'University', 'in', '1967', '.'], entities=[Entity(span=(0, 2), label='Entity'), Entity(span=(6, 8), label='Entity'), Entity(span=(4, 5), label='Entity'), Entity(span=(9, 10), label='Entity')], relations=[Relation(head=(0, 2), tail=(6, 8), label='educated at', qualifiers=[Entity(span=(4, 5), label='academic degree'), Entity(span=(9, 10), label='end time')])], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['if', 'you', 'come', 'to', 'Finland', 'then', 'you', 'need', 'a', 'residence', 'permit', 'and', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['At', '*@UNK@*', 'you', 'can', 'complete', 'a', '*@UNK@*', 'acquire', 'new', 'skills', 'or', 'supplement', '*@UNK@*', 'already', 'acquired', 'skills', '-', 'you', 'can', 'train', 'and', 'find', 'a', 'new', 'direction', 'for', '*@UNK@*', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['Espoo', 'Upper', 'Secondary', 'School', 'for', '*@UNK@*', 'is', 'full', 'of', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['If', 'you', 'are', 'already', 'in', '*@UNK@*', 'book', 'an', 'appointment', 'at', 'a', 'service', 'point', 'of', 'the', 'Finnish', 'Immigration', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['If', 'you', 'get', 'a', 'positive', '*@UNK@*', 'you', 'will', 'get', 'a', 'residence', 'permit', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['If', 'you', 'are', 'a', 'private', '*@UNK@*', 'meaning', 'that', 'you', 'have', 'an', '*@UNK@*', 'business', '*@UNK@*', 'getting', 'a', 'Business', '*@UNK@*', 'requires', 'that', '*@UNK@*', 'domicile', 'is', 'registered', 'in', 'an', '*@UNK@*', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['You', 'have', 'a', '*@UNK@*', 'pursue', 'a', 'trade', 'or', 'engage', 'in', 'business', 'activities', 'in', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['The', '*@UNK@*', 'Office', 'sends', 'each', 'person', 'being', 'transferred', 'to', 'the', 'pilot', 'a', 'letter', 'or', 'message', 'in', 'the', '*@UNK@*'], entities=[Entity(span=(1, 2), label='Entity'), Entity(span=(17, 18), label='Entity'), Entity(span=(10, 11), label='Entity')], relations=[Relation(head=(1, 2), tail=(17, 18), label='subclass of', qualifiers=[Entity(span=(10, 11), label='of')])], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None), Sentence(tokens=['At', 'the', 'clinic', '*@UNK@*', 'you', 'can', 'review', 'funding', 'models', 'and', 'discuss', 'possible', '*@UNK@*', '*@UNK@*'], entities=[], relations=[], wordpieceSentText=None, wordpieceTokensIndex=None, wordpieceSegmentIds=None, jointLabelMatrix=None, quintupletMatrix=None)]
        file = open("tmp.json", "r")
        raw = file.read()
        raw = json.loads(raw)
        print("raw", raw, flush=True)

        for i in range (0,len(raw['relations'])):
            print('ENTERING LOOP')
            relation = raw['relations'][i]['label']
            entity1_head = raw['relations'][i]['head'][0]
            entity1_tail = raw['relations'][i]['head'][1]
            entity2_head = raw['relations'][i]['tail'][0]
            entity2_tail = raw['relations'][i]['tail'][1]
            qualifiers_label = raw['relations'][i]['qualifiers'][0]['label']
            qualifiers_value = split_text[raw['relations'][i]['qualifiers'][0]['span'][0]:raw['relations'][i]['qualifiers'][0]['span'][1]]
            entity1 = split_text[entity1_head:entity1_tail]
            entity2 = split_text[entity2_head:entity2_tail]
            print(relation, entity1,entity2, [qualifiers_label, qualifiers_value] ,flush=True)
            RE_row = [[url, text, entity1, relation, entity2, [qualifiers_label,qualifiers_value] ]]
            RE_row = pd.DataFrame( RE_row,
                columns=['url', 'input_text', 'Entity1', 'Relationship', 'Entity2', 'Qualifiers'])
            print('**** Write row to df ****', flush=True)
#             RE_row.to_csv('data_RE_text2kg.csv', sep=";",mode='a', header=False
            RE_row.to_csv('data_RE.csv', mode='a', header=False)

    except Exception as e:
        print(e)


# data_cube_RE.to_csv('data_RE')

