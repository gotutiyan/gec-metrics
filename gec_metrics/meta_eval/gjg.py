import glob
import os
from gec_metrics.metrics.base import MetricBase
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
from .base import MetaEvalBase

class MetaEvalGJG(MetaEvalBase):
    @dataclass
    class GJGOutput(MetaEvalBase.Output):
        '''The dataclass to store the meta-evaluation results.
        
        Args:
            ts (MetaEvalBase.Corr):
                The correlation using TrueSkill-based human evaluation.
            ts (MetaEvalBase.Corr):
                The correlation using Expected Wins-based human evaluation.
        '''
        ts: MetaEvalBase.Corr = None
        ew: MetaEvalBase.Corr = None

    def __init__(self, config: MetaEvalBase.Config):
        super().__init__(config)
        self.system_data = self.load_system_data()

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
        data_dir = glob.glob('**/conll14/', recursive=True)[0]
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
            sents = open(os.path.join(data_dir, 'official_submissions', model)).read().rstrip().split('\n')
            sentences.append(sents)
        data['hypotheses'] = sentences
        input_sents = open(os.path.join(data_dir, 'official_submissions', 'INPUT')).read().rstrip().split('\n')
        data['sources'] = input_sents
        ref0 = open(os.path.join(data_dir, 'REF0')).read().rstrip().split('\n')
        ref1 = open(os.path.join(data_dir, 'REF1')).read().rstrip().split('\n')
        data['references'] = [ref0, ref1]
        return data
    
    def load_sentence_data(self) -> dict[str, list]:
        '''TODO: Implement it.'''
        return super().load_sentence_data()
    
    def corr_system(self, metric: MetricBase) -> "GJGOutput":
        '''Compute system-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            GJGOutput: The correlations.
        '''
        data = self.system_data
        my_scores = []
        for model_id, model in enumerate(data['models']):
            score = self.calc_system_score(
                metric,
                data['sources'],
                data['hypotheses'][model_id],
                data['references']
            )
            my_scores.append(score)
        corrs = []
        for score_id in ['ew', 'ts']:
            corr = self.Corr()
            corr.pearson = pearsonr(my_scores, data['human_score'][score_id])[0].item()
            corr.spearman = spearmanr(my_scores, data['human_score'][score_id])[0].item()
            corrs.append(corr)
        return self.GJGOutput(
            ew=corrs[0],
            ts=corrs[1]
        )
        
    def corr_sentence(self, scorer: MetricBase):
        '''TODO: Implement it.'''
        return super().corr_sentence(scorer)