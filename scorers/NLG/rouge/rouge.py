import numpy as np
import torch.nn as nn
from rouge_score import rouge_scorer
from six.moves import zip_longest


class Rouge(nn.Module):
    def __init__(self, rouges, **kwargs):
        super().__init__()
        rouges = [r.replace("rougel", "rougeL") for r in rouges]
        self.scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
        self.rouges = rouges

    def forward(self, refs, hyps):
        scores = []
        for target_rec, prediction_rec in zip_longest(refs, hyps):
            if target_rec is None or prediction_rec is None:
                raise ValueError("Must have equal number of lines across target and " "prediction.")
            scores.append(self.scorer.score(target_rec, prediction_rec))
        f1_rouge = [s[self.rouges[0]].fmeasure for s in scores]
        return np.mean(f1_rouge), f1_rouge


class Rouge1(Rouge):
    def __init__(self, **kwargs):
        super(Rouge1, self).__init__(rouges=["rouge1"])


class Rouge2(Rouge):
    def __init__(self, **kwargs):
        super(Rouge2, self).__init__(rouges=["rouge2"])


class RougeL(Rouge):
    def __init__(self, **kwargs):
        super(RougeL, self).__init__(rouges=["rougeL"])


if __name__ == "__main__":

    hyps = ["nothing to do lol", "nothing to do x", "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax."]
    refs = ["heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.", "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.", "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities."]
    score = RougeL()(refs, hyps)[0]
    print(score)
