import glob
import os
from scipy.stats import pearsonr, spearmanr
from gec_metrics.metrics import MetricBaseForReferenceBased
from dataclasses import dataclass

@dataclass
class Corr:
    pearson: float = None
    spearman: float = None

@dataclass
class GJGSystemCorrOutput:
    ew: Corr = None
    ts: Corr = None

@dataclass
class Score:
    accuracy: float = None
    kendall: float = None
    
@dataclass
class GJGSentenceCorrOutput:
    ew: Score = None
    ts: Score = None

def gjg_load_data_for_system_corr():
    # Expected Wins scores
    ew_table = '''0.628 1 AMU
0.566 2-3 RAC
0.561 2-4 CAMB
0.550 3-5 CUUI
0.539 4-5 POST
0.513 6-8 UFC
0.506 6-8 PKU
0.495 7-9 UMC
0.485 7-10 IITB
0.463 10-11 SJTU
0.456 9-12 INPUT
0.437 11-12 NTHU
0.300 13 IPN'''.split('\n')
    ts_table = '''0.273 1 AMU
0.182 2 CAMB
0.114 3-4 RAC
0.105 3-5 CUUI
0.080 4-5 POST
-0.001 6-7 PKU
-0.022 6-8 UMC
-0.041 7-10 UFC
-0.055 8-11 IITB
-0.062 8-11 INPUT
-0.074 9-11 SJTU
-0.142 12 NTHU
-0.358 13 IPN'''.split('\n')
    
    ew_models = [line.split(' ')[2] for line in ew_table]
    ew_scores = [float(line.split(' ')[0]) for line in ew_table]
    ts_models = [line.split(' ')[2] for line in ts_table]
    ts_scores = [float(line.split(' ')[0]) for line in ts_table]
    ts_scores_reorder = [ts_scores[ts_models.index(m)] for m in ew_models]
    data_dir = glob.glob('**/conll14/', recursive=True)[0]
    data = {
        'sentences': [],
        'references': [],
        'human_score': {
            'ew': ew_scores,
            'ts': ts_scores_reorder
        },
        'models': ew_models,
        'input_sentences': []
    }
    sentences = []
    for model in ew_models:
        sents = open(os.path.join(data_dir, 'official_submissions', model)).read().rstrip().split('\n')
        sentences.append(sents)
    data['sentences'] = sentences
    input_sents = open(os.path.join(data_dir, 'official_submissions', 'INPUT')).read().rstrip().split('\n')
    data['input_sentences'] = input_sents
    ref0 = open(os.path.join(data_dir, 'REF0')).read().rstrip().split('\n')
    ref1 = open(os.path.join(data_dir, 'REF1')).read().rstrip().split('\n')
    data['references'] = [ref0, ref1]
    return data

def gjg_system_corr(scorer):
    data = gjg_load_data_for_system_corr()
    my_scores = []
    for model_id, model in enumerate(data['models']):
        if isinstance(scorer, MetricBaseForReferenceBased):
            score = scorer.score_corpus(
                sources=data['input_sentences'],
                hypotheses=data['sentences'][model_id],
                references=data['references']
            )
        else:
            score = scorer.score_corpus(
                sources=data['input_sentences'],
                hypotheses=data['sentences'][model_id]
            )
        my_scores.append(score)
    corrs = []
    for score_id in ['ew', 'ts']:
        corr = Corr()
        corr.pearson = pearsonr(my_scores, data['human_score'][score_id])[0]
        corr.spearman = spearmanr(my_scores, data['human_score'][score_id])[0]
        corrs.append(corr)
    return GJGSystemCorrOutput(
        ew=corrs[0],
        ts=corrs[1]
    )