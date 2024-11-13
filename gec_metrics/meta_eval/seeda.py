import argparse
import glob
import os
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
import itertools
from .base import MetaEvalBase
from gec_metrics.metrics import MetricBase
import xml.etree.ElementTree as ET
import numpy as np
from .utils import read_lines

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
        ew_edit: MetaEvalBase.Corr = None
        ew_sent: MetaEvalBase.Corr = None
        ts_edit: MetaEvalBase.Corr = None
        ts_sent: MetaEvalBase.Corr = None

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
            sents = read_lines(os.path.join(subset_dir, model + '.txt'))
            data['hypotheses'].append(sents)
        
        score_dir = glob.glob('**/SEEDA/scores/human', recursive=True)[0]
        for score_id in self.SCORE_ID:
            scores = list(map(float, read_lines(
                os.path.join(score_dir, score_id + '.txt')
            )))
            scores = [s for i, s in enumerate(scores) if self.MODELS[i] not in del_systems]
            data['human_score'][score_id] = scores

        data['sources'] = read_lines(os.path.join(subset_dir, 'INPUT.txt'))

        ref0 = read_lines(os.path.join(subset_dir, 'REF0.txt'))
        ref1 = read_lines(os.path.join(subset_dir, 'REF1.txt'))
        data['references'] = [ref0, ref1]
        return data
    
    def load_xml(self, xml_path: str, target_models: list[str]):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # The human scores will be 
        #   human_scores[src_id][annotation_id][sys_id].
        # Beucase multiple annotations may exist in the same sentence.
        # Each element contains None or a minus rank of the system.
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
                    # Put the minus ranking as a score
                    scores[target_models.index(sys)] = -rank
            human_scores[src_id].append(scores)
        # Sort by source id.
        human_scores = sorted(human_scores.items(), key=lambda x:x[0])
        human_scores = [h[1] for h in human_scores]
        return human_scores
    
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
            'human_score': dict(),
            'human_score_paths': dict(),
            'models': models,
            'del_models': ['INPUT', 'REF-F', 'GPT-3.5'],
            'sources': []
        }
        data['human_score']['edit'] = self.load_xml(
            data_dir + 'judgments_edit.xml',
            models
        )
        data['human_score']['sent'] = self.load_xml(
            data_dir + 'judgments_sent.xml',
            models
        )
        for model in models:
            sents = read_lines(os.path.join(subset_dir, model + '.txt'))
            data['hypotheses'].append(sents)
        
        input_sents = read_lines(os.path.join(subset_dir, 'INPUT.txt'))
        data['sources'] = input_sents

        ref0 = read_lines(os.path.join(subset_dir, 'REF0.txt'))
        ref1 = read_lines(os.path.join(subset_dir, 'REF1.txt'))
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
            ) for name in self.SCORE_ID
        ]
        return self.SEEDASystemCorrOutput(
            ew_edit=corrs[0],
            ew_sent=corrs[1],
            ts_edit=corrs[2],
            ts_sent=corrs[3]
        )
    
    def corr_sentence(
        self, metric: MetricBase
    ) -> "SEEDASentenceCorrOutput":
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
        corrs = []
        num_sents = len(data['sources'])
        num_sys = len(data['models'])
        for name in ['edit', 'sent']:
            human_scores = data['human_score'][name]
            agree = 0
            not_agree = 0
            denominator = 0

            for src_id in range(num_sents):
                for annotate_id in range(len(human_scores[src_id])):
                    for sys1, sys2 in itertools.combinations(range(num_sys), 2):
                        m1 = metric_scores[sys1][src_id]
                        m2 = metric_scores[sys2][src_id]
                        # The human rank is minus ranking value,
                        #   so higher values indicate higher quality.
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
            corrs.append(corr)
        return self.SEEDASentenceCorrOutput(
            edit=corrs[0],
            sent=corrs[1]
        )