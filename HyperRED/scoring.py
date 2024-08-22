
from typing import Dict, List, Tuple

from data_process import RawPred, Sentence


def safe_divide(a: float, b: float) -> float:
    if a == 0.0 or b == 0.0:
        return 0.0
    return a / b


class Scorer:
    name: str = ""

    def run(self, pred: List[Sentence], gold: List[Sentence]) -> Dict[str, float]:
        raise NotImplementedError


class StrictScorer:
    name: str = "strict triplet"

    def make_sent_tuples(
        self, s: Sentence
    ) -> List[Tuple[Tuple[int, int, str], Tuple[int, int, str], str]]:
        id_to_entity = {e.span: e for e in s.entities}
        tuples = []
        for r in s.relations:
            head = id_to_entity[r.head]
            tail = id_to_entity[r.tail]
            t = (
                (head.span[0], head.span[1], head.label),
                (tail.span[0], tail.span[1], tail.label),
                r.label,
            )
            tuples.append(t)
        return tuples

    def match_gold_to_pred(
        self, pred: List[Sentence], gold: List[Sentence]
    ) -> List[Sentence]:
        assert self is not None
        text_to_pred = {p.text: p for p in pred}
        empty = RawPred.empty().as_sentence(None)
        matched = [text_to_pred.get(s.text, empty) for s in gold]
        print("\nHow many gold sents have no matching pred?")
        print(dict(num=len([p for p in matched if p == empty])))
        return matched

    def run(self, pred: List[Sentence], gold: List[Sentence]) -> Dict[str, float]:
        pred = self.match_gold_to_pred(pred, gold)
        assert len(pred) == len(gold)
        num_correct = 0
        num_pred = 0
        num_gold = 0

        for p, g in zip(pred, gold):
            tuples_pred = self.make_sent_tuples(p)
            tuples_gold = self.make_sent_tuples(g)
            num_pred += len(tuples_pred)
            num_gold += len(tuples_gold)
            for a in tuples_pred:
                for b in tuples_gold:
                    if a == b:
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        return dict(
            num_correct=num_correct,
            num_pred=num_pred,
            num_gold=num_gold,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class EntityScorer(StrictScorer):
    name: str = "entity"

    def make_sent_tuples(self, s: Sentence) -> List[Tuple[int, int, str]]:
        tuples = [(e.span[0], e.span[1], e.label) for e in s.entities]
        return sorted(set(tuples))


class QuintupletScorer(StrictScorer):
    name: str = "quintuplet"

    def make_sent_tuples(
        self, s: Sentence
    ) -> List[Tuple[int, int, int, int, int, int, str, str]]:
        tuples = []
        for r in s.relations:
            for q in r.qualifiers:
                t = (
                    r.head[0],
                    r.head[1],
                    r.tail[0],
                    r.tail[1],
                    q.span[0],
                    q.span[1],
                    r.label,
                    q.label,
                )
                tuples.append(t)
        return tuples