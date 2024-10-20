import argparse
import glob
import os
from scipy.stats import pearsonr, spearmanr
from meta_eval_data.SEEDA.utils.corr_sentence import make_h_mtx, calc_corr
from dataclasses import dataclass
import itertools
from .base import MetaEvalBase
from gec_metrics.metrics import MetricBase

# This function is based on the official one (https://github.com/tmu-nlp/SEEDA),
#   but fixed to input score as list[list[float]], not file paths.
def make_m_mtx(
    metric_score: list[list[float]],
    targets: list[str],
    order: str = 'higher'
):
    m_mtx = {}
    N = len(targets)
    src_ids = [11, 12, 28, 35, 36, 45, 46, 52, 54, 58, 62, 64, 65, 68, 70, 73, 75, 76, 85, 87, 93, 96, 98, 105, 111, 115, 116, 118, 125, 130, 131, 132, 133, 136, 138, 143, 145, 155, 157, 160, 162, 163, 168, 170, 173, 176, 178, 179, 180, 197, 198, 200, 201, 202, 205, 206, 207, 209, 212, 213, 219, 226, 228, 229, 231, 233, 236, 240, 243, 247, 252, 254, 258, 260, 261, 266, 268, 270, 272, 274, 282, 284, 288, 295, 300, 302, 304, 310, 312, 313, 315, 316, 318, 319, 321, 327, 329, 336, 339, 349, 351, 355, 363, 371, 373, 380, 383, 389, 392, 395, 396, 398, 400, 401, 402, 407, 415, 424, 426, 428, 431, 436, 441, 446, 447, 449, 450, 451, 452, 454, 455, 458, 461, 471, 474, 476, 485, 488, 490, 494, 501, 502, 504, 505, 506, 507, 510, 511, 521, 525, 528, 536, 537, 538, 539, 552, 553, 558, 579, 590, 592, 595, 597, 603, 606, 609, 610, 614, 615, 620, 623, 625, 627, 631, 636, 639, 645, 646, 647, 648, 649, 651, 654, 658, 664, 665, 668, 673, 674, 675, 678, 681, 682, 683, 686, 697, 700, 701, 702, 705, 707, 710, 711, 715, 717, 719, 720, 724, 725, 726, 727, 730, 731, 734, 743, 744, 745, 749, 754, 756, 759, 765, 766, 769, 771, 772, 777, 778, 779, 780, 781, 782, 783, 784, 787, 788, 791, 792, 796, 797, 799, 806, 808, 810, 811, 812, 817, 818, 823, 824, 825, 827, 828, 830, 835, 837, 838, 842, 844, 846, 849, 851, 853, 856, 857, 859, 867, 872, 873, 875, 881, 886, 887, 888, 889, 891, 895, 901, 902, 905, 910, 911, 912, 913, 916, 918, 923, 924, 930, 933, 936, 938, 942, 943, 950, 961, 962, 963, 968, 973, 974, 977, 979, 982, 998, 999, 1001, 1003, 1012, 1016, 1017, 1024, 1031, 1038, 1040, 1051, 1052, 1055, 1056, 1057, 1063, 1064, 1065, 1069, 1070, 1075, 1081, 1083, 1084, 1085, 1088, 1089, 1090, 1091, 1099, 1100, 1104, 1106, 1109, 1110, 1111, 1112, 1119, 1128, 1140, 1142, 1143, 1145, 1146, 1151, 1152, 1154, 1160, 1162, 1168, 1170, 1176, 1184, 1195, 1196, 1202, 1205, 1206, 1207, 1209, 1211, 1212, 1213, 1222, 1224, 1229, 1231, 1234, 1245, 1247, 1249, 1260, 1261, 1263, 1264, 1265, 1275, 1288, 1290, 1294, 1297, 1298, 1300, 1303, 1306, 1310]
    
    set = metric_score

    scores = []
    for i in range(len(src_ids)):
        score = []
        for j in range(N):
            score.append(set[j][i])
        scores.append(score)

    i = 0
    for src_id, score in zip(src_ids, scores):
        sub_mtx = [[None for _ in range(N)] for _ in range(N)]
        comb = itertools.combinations(range(N), 2)
        for c in comb:
            t1 = c[0]
            t2 = c[1]
            score1 = scores[i][t1]
            score2 = scores[i][t2]
            if score1 > score2:
                if order == 'higher':
                    sub_mtx[t1][t2] = 1
                else:
                    sub_mtx[t1][t2] = -1
            else:
                if order == 'higher':
                    sub_mtx[t1][t2] = -1
                else:
                    sub_mtx[t1][t2] = 1
        m_mtx[str(src_id)] = sub_mtx
        i += 1

    m_mtx = {k: m_mtx[k] for k in sorted(m_mtx.keys())}
    return m_mtx

class MetaEvalSEEDA(MetaEvalBase):
    MODELS = ['BART', 'BERT-fuse', 'GECToR-BERT', 'GECToR-ens', 'GPT-3.5', 'INPUT', 'LM-Critic', 'PIE', 'REF-F', 'REF-M', 'Riken-Tohoku', 'T5', 'TemplateGEC', 'TransGEC', 'UEDIN-MS']
    SCORE_ID = ['EW_edit', 'EW_sent', 'TS_edit', 'TS_sent']
    @dataclass
    class SEEDASystemCorrOutput(MetaEvalBase.Output):
        '''The dataclass to store system-level correlations.

        Args:
            ew_sent (MetaEvalBase.Corr):
                SEEDA-S correlation based on Expected Wins-based human evaluation.
            ew_edit (MetaEvalBase.Corr):
                SEEDA-E correlation based on Expected Wins-based human evaluation.
            ts_sent (MetaEvalBase.Corr):
                SEEDA-S correlation based on TrueSkill-based human evaluation.
            ts_edit (MetaEvalBase.Corr):
                SEEDA-E correlation based on TrueSkill-based human evaluation.
        '''
        ew_sent: MetaEvalBase.Corr = None
        ew_edit: MetaEvalBase.Corr = None
        ts_sent: MetaEvalBase.Corr = None
        ts_edit: MetaEvalBase.Corr = None

    @dataclass
    class SEEDASentenceCorrOutput(MetaEvalBase.Output):
        '''The dataclass to store sentence-level correlations.

        Args:
            sent (MetaEvalBase.Corr):
                SEEDA-S sentence-level correlation.
            edit (MetaEvalBase.Corr):
                SEEDA-E sentence-level correlation.
        '''
        sent: MetaEvalBase.Corr = None
        edit: MetaEvalBase.Corr = None

    @dataclass
    class Config:
        system: str = 'base'

    def __init__(self, config: MetaEvalBase.Config):
        super().__init__(config)
        self.system_data = self.load_system_data()
        self.sentence_data = self.load_sentence_data()

    def load_system_data(self) -> dict[str, list]:
        '''Load evaluation data and human scores for system-level meta-evalaution.
        '''
        subset_dir = glob.glob('**/SEEDA/outputs/subset', recursive=True)[0]
        del_systems = {
            'base': ['INPUT', 'REF-F', 'GPT-3.5'],
            '+INPUT': ['REF-F', 'GPT-3.5'],
            '+REF-F_GPT-3.5': ['INPUT'],
            'all': []
        }[self.config.system]
        models = [m for m in self.MODELS if m not in del_systems]
        data = {
            'hypotheses': [],
            'references': [],
            'human_score': dict(),
            'models': models,
            'sources': []
        }
        for model in models:
            sents = open(os.path.join(subset_dir, model + '.txt')).read().rstrip().split('\n')
            data['hypotheses'].append(sents)
        
        score_dir = glob.glob('**/SEEDA/scores/human', recursive=True)[0]
        for score_id in self.SCORE_ID:
            scores = list(map(float, open(
                os.path.join(score_dir, score_id + '.txt')
            ).read().rstrip().split('\n')))
            scores = [s for i, s in enumerate(scores) if self.MODELS[i] not in del_systems]
            data['human_score'][score_id] = scores

        data['sources'] = open(os.path.join(subset_dir, 'INPUT.txt')).read().rstrip().split('\n')

        ref0 = open(os.path.join(subset_dir, 'REF0.txt')).read().rstrip().split('\n')
        ref1 = open(os.path.join(subset_dir, 'REF1.txt')).read().rstrip().split('\n')
        data['references'] = [ref0, ref1]
        return data
    
    def load_sentence_data(self) -> dict[str, int]:
        '''Load evaluation data and human scores for sentence-level meta-evaluation.'''
        subset_dir = glob.glob('**/SEEDA/outputs/subset/', recursive=True)[0]
        data_dir = glob.glob('**/SEEDA/data/', recursive=True)[0]
        del_systems = {
            'base': ['INPUT', 'REF-F', 'GPT-3.5'],
            '+INPUT': ['REF-F', 'GPT-3.5'],
            '+REF-F_GPT-3.5': ['INPUT'],
            'all': []
        }[self.config.system]
        del_systems += ['REF0', 'REF1']
        models = [m for m in self.MODELS if m not in del_systems]
        data = {
            'hypotheses': [],
            'human_score_paths': dict(),
            'models': models,
            'del_models': ['INPUT', 'REF-F', 'GPT-3.5'],
            'sources': []
        }
        data['human_score_paths']['edit'] = glob.glob(data_dir + 'judgments_edit.xml')[0]
        data['human_score_paths']['sent'] = glob.glob(data_dir + 'judgments_sent.xml')[0]
        for model in models:
            sents = open(os.path.join(subset_dir, model + '.txt')).read().rstrip().split('\n')
            data['hypotheses'].append(sents)
        
        input_sents = open(os.path.join(subset_dir, 'INPUT.txt')).read().rstrip().split('\n')
        data['sources'] = input_sents

        ref0 = open(os.path.join(subset_dir, 'REF0.txt')).read().rstrip().split('\n')
        ref1 = open(os.path.join(subset_dir, 'REF1.txt')).read().rstrip().split('\n')
        data['references'] = [ref0, ref1]
        return data
    
    def corr_system(self, metric: MetricBase) -> "SEEDASystemCorrOutput":
        '''Compute system-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            SEEDASystemCorrOutput: The correlations.
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
        for score_id in self.SCORE_ID:
            corr = self.Corr()
            corr.pearson = pearsonr(my_scores, data['human_score'][score_id])[0].item()
            corr.spearman = spearmanr(my_scores, data['human_score'][score_id])[0].item()
            corrs.append(corr)
        return self.SEEDASystemCorrOutput(
            ew_edit=corrs[0],
            ew_sent=corrs[1],
            ts_edit=corrs[2],
            ts_sent=corrs[3],
        )
    
    def corr_sentence(self, metric: MetricBase) -> 'SEEDASentenceCorrOutput':
        '''Compute sentence-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            SEEDASentenceCorrOutput: The correlations.
        '''
        data = self.sentence_data
        metric_score = []
        for model_id, model in enumerate(data['models']):
            scores = self.calc_sentence_score(
                metric,
                data['sources'],
                data['hypotheses'][model_id],
                data['references']
            )
            metric_score.append(scores)
            
        corrs = []
        for mode in ['edit', 'sent']:
            h_mtx = make_h_mtx(data['human_score_paths'][mode], data['models'], data['del_models'])
            m_mtx = make_m_mtx(metric_score, data['models'])
            acc, ken = calc_corr(h_mtx, m_mtx)
            score = self.Corr(accuracy=acc, kendall=ken)
            corrs.append(score)
        return self.SEEDASentenceCorrOutput(
            edit=corrs[0],
            sent=corrs[1]
        )