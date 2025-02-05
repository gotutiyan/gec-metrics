import glob
import os
from gec_metrics.metrics.base import MetricBase
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
from .base import MetaEvalBase
import xml.etree.ElementTree as ET
import itertools
from .utils import read_lines

class MetaEvalGJG(MetaEvalBase):
    MODELS = ['AMU', 'RAC', 'CAMB', 'CUUI', 'POST', 'UFC', 'PKU', 'UMC', 'IITB', 'SJTU', 'INPUT', 'NTHU', 'IPN']
    @dataclass
    class GJGSystemCorrOutput(MetaEvalBase.Output):
        '''The dataclass to store the meta-evaluation results.
        
        Args:
            ts (MetaEvalBase.Corr):
                The correlation using TrueSkill-based human evaluation.
            ts (MetaEvalBase.Corr):
                The correlation using Expected Wins-based human evaluation.
        '''
        ew: MetaEvalBase.Corr = None
        ts: MetaEvalBase.Corr = None

    @dataclass
    class GJGSentenceCorrOutput(MetaEvalBase.Output):
        '''The dataclass to store the meta-evaluation results.
        
        Args:
            ts (MetaEvalBase.Corr):
                The correlation using TrueSkill-based human evaluation.
            ts (MetaEvalBase.Corr):
                The correlation using Expected Wins-based human evaluation.
        '''
        corr: MetaEvalBase.Corr = None

    def __init__(self, config: MetaEvalBase.Config = None):
        super().__init__(config)
        self.system_data = self.load_system_data()
        self.sentence_data = self.load_sentence_data()

    def load_system_data(self) -> dict[str, list]:
        '''Loads evaluation data and human scores.'''
        # Expected Wins scores
        # Table 3 (b) https://aclanthology.org/D15-1052.pdf
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
        # TrueSkill scores
        # Table 3 (c) https://aclanthology.org/D15-1052.pdf
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
        data_dir = glob.glob('**/meta_eval_data/conll14/', recursive=True)[0]
        data = {
            'hypotheses': [],
            'references': [],
            'human_score': {
                'ew': ew_scores,
                'ts': ts_scores_reorder
            },
            'models': ew_models,
            'sources': []
        }
        sentences = []
        for model in ew_models:
            sents = read_lines(os.path.join(data_dir, 'official_submissions', model))
            sentences.append(sents)
        data['hypotheses'] = sentences
        input_sents = read_lines(os.path.join(data_dir, 'official_submissions', 'INPUT'))
        data['sources'] = input_sents
        ref0 = read_lines(os.path.join(data_dir, 'REF0'))
        ref1 = read_lines(os.path.join(data_dir, 'REF1'))
        data['references'] = [ref0, ref1]
        return data
    
    def load_xml(self, xml_path: str, target_models: list[str]):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        human_scores = dict()
        for child in root.find('error-correction-ranking-result'):
            src_id = int(child.attrib['src-id'])
            human_scores[src_id] = human_scores.get(
                src_id, []
            )
            scores = [None] * len(target_models)
            for trans in child:
                systems = trans.attrib['system'].split()
                rank = int(trans.attrib['rank'])
                for sys in systems:
                    if sys not in target_models:
                        continue
                    # Put the minus ranking as the score
                    scores[target_models.index(sys)] = -rank
            human_scores[src_id].append(scores)
        human_scores = sorted(human_scores.items(), key=lambda x:x[0])
        return human_scores
    
    def load_sentence_data(self) -> dict[str, list]:
        '''TODO: Implement it.'''
        data_dir = glob.glob('**/meta_eval_data/conll14/', recursive=True)[0]
        score_dir = glob.glob('**/meta_eval_data/GJG15/', recursive=True)[0]
        data = {
            'hypotheses': [],
            'references': [],
            'human_score': [],
            'models': self.MODELS,
            'sources': []
        }
        data['human_score'] = self.load_xml(os.path.join(score_dir, 'judgments.xml'), self.MODELS)
        src_ids = [h[0] for h in data['human_score']]
        data['human_score'] = [h[1] for h in data['human_score']]
        sentences = []
        for model in self.MODELS:
            sents = read_lines(os.path.join(data_dir, 'official_submissions', model))
            sentences.append([sents[i] for i in src_ids])
        data['hypotheses'] = sentences
        input_sents = read_lines(os.path.join(data_dir, 'official_submissions', 'INPUT'))
        data['sources'] = [input_sents[i] for i in src_ids]
        ref0 = read_lines(os.path.join(data_dir, 'REF0'))
        ref1 = read_lines(os.path.join(data_dir, 'REF1'))
        data['references'] = [
            [ref0[i] for i in src_ids],
            [ref1[i] for i in src_ids],
        ]

        return data
    
    def corr_system(self, metric: MetricBase) -> "GJGSystemCorrOutput":
        '''Compute system-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            GJGOutput: The correlations.
        '''
        data = self.system_data
        metric_scores = [
            self.calc_system_score(
                metric=metric,
                sources=data['sources'],
                hypotheses=hyps,
                references=data['references']
            ) for hyps in data['hypotheses']
        ]
        corrs = [
            self.Corr(
                pearson=float(pearsonr(metric_scores, data['human_score'][name])[0]),
                spearman=float(spearmanr(metric_scores, data['human_score'][name])[0])
            ) for name in ['ew', 'ts']
        ]
        return self.GJGSystemCorrOutput(
            ew=corrs[0],
            ts=corrs[1]
        )
        
    def corr_sentence(self, metric: MetricBase) -> "GJGSentenceCorrOutput":
        '''Compute sentence-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            SEEDASentenceCorrOutput: The correlations.
        '''
        data = self.sentence_data
        metric_scores = [
            self.calc_sentence_score(
                metric=metric,
                sources=data['sources'],
                hypotheses=hyps,
                references=data['references']
            ) for hyps in data['hypotheses']
        ]
        num_sents = len(data['sources'])
        num_sys = len(data['models'])
        human_scores = data['human_score']
        agree = 0
        not_agree = 0
        denominator = 0
        for src_id in range(num_sents):
            for annotate_id in range(len(human_scores[src_id])):
                for sys1, sys2 in itertools.combinations(range(num_sys), 2):
                    m1 = metric_scores[sys1][src_id]
                    m2 = metric_scores[sys2][src_id]
                    h1 = human_scores[src_id][annotate_id][sys1]
                    h2 = human_scores[src_id][annotate_id][sys2]
                    if None in [h1, h2]:
                        continue
                    if h1 == h2:
                        continue
                    denominator += 1
                    if (m1 <= m2) == (h1 <= h2):
                        agree += 1
                    else:
                        not_agree += 1
        corr = self.Corr()
        corr.accuracy = agree / denominator
        corr.kendall = (agree - not_agree) / denominator
        return self.GJGSentenceCorrOutput(
            corr=corr
        )