import abc
from dataclasses import dataclass
from gec_metrics.metrics import (
    MetricBase,
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForSourceFree,
    inputs_handler
)
from gec_metrics.metrics.llm_kobayashi24 import LLMKobayashi24
import itertools
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MetaEvalBase(abc.ABC):
    @dataclass
    class Config: ...

    @dataclass
    class Corr:
        pearson: float = None
        spearman: float = None
        accuracy: float = None
        kendall: float = None
        human_scores: list[float] = None
        metric_scores: list[float] = None

    @dataclass
    class Output: ...

    def __init__(self, config: Config = None):
        self.config = config if config is not None else self.Config()
        self.system_data = None
        self.sentence_data = None
    
    @abc.abstractmethod
    def load_system_data(self) -> dict[str, list]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_sentence_data(self) -> dict[str, list]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def corr_system(
        self,
        metric: MetricBase,
        aggregation: str='default'
    ):
        '''Compute system-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.
            aggregation (str): 
                How to aggregate sentence-level scores into system rankings.
                - 'default': Default aggregation, e.g.,average or accumulation.
                - 'trueskill': TrueSkill aggregation.

        Returns:
            **SystemCorrOutput: The system-level correlations output.
        '''
        data = self.system_data
        metric_scores = metric.rank_systems(
            **inputs_handler(
                metric, data['sources'], data['hypotheses'], data['references']
            ),
            aggregation=aggregation
        )
        corrs = [
            self.Corr(
                pearson=float(pearsonr(metric_scores, data['human_score'][name])[0]),
                spearman=float(spearmanr(metric_scores, data['human_score'][name])[0]),
                human_scores=data['human_score'][name],
                metric_scores=metric_scores
            ) for name in self.SCORE_ID
        ]
        return corrs
    
    def rearange_sent_data(self, data):
        '''Rearange the format and content of sentence-level evaluation results.
        This is intednded to use for LLMKobayashi24** metric.
        The LLMKobayashi24** sentence-level meta evaluation requires the input 
            of the same set of corrected sentences as in the SEEDA manual evaluation.
        On the other hand, when the number of corrected sentences is larger than 5,
            the same sentences as in SEEDA are not necessarily sampled and evaluated.
        To ensure that the same sentences as in SEEDA are used in the evaluation,
            this function replaces the unused sentences with the used ones.
        This ensures that only the same sentences as SEEDA are present in the hypothesis set.

        Also, different annotators evaluate different subsets of the same hypothesis set,
            so we flatten the data to make this easier to handle.
        Specifically, human scores will be changed 
            from [num_sentences][num_annotatinos][num_systems] to [num_sentences * num_annotations][1][num_systems].
        The hypotheses also be expanded: [num_systems][num_sentences] -> [num_systems][num_sentences * num_annotations].
        '''

        human_score = data['human_score']
        human_aspects = list(human_score.keys())
        hypotheses = data['hypotheses']
        references = data['references']
        num_systems = len(hypotheses)
        num_sents = len(hypotheses[0])
        num_refs = len(references)
        flatten_data = {
            'sources': [],
            'hypotheses': [[] for _ in range(num_systems)],
            'references': [[] for _ in range(num_refs)],
            'human_score': {k: [] for k in data['human_score'].keys()},
            'models': data['models']
        }
        for sent_id in range(num_sents):
            for ann_id in range(len(human_score[human_aspects[0]][sent_id])):
                # this rank has minus rank or None. None means not evaluated in human evaluation.
                this_h_score = human_score[human_aspects[0]][sent_id][ann_id]
                if all([s is None for s in this_h_score]):
                    # GJG15 results sometimes empty
                    continue
                # Classify the systems by checking the element is None or not.
                unused_model_ids = [i for i, s in enumerate(this_h_score) if s is None]
                used_model_ids = [i for i, s in enumerate(this_h_score) if s is not None]

                flatten_data['sources'].append(data['sources'][sent_id])
                for ref_id in range(num_refs):
                    flatten_data['references'][ref_id].append(references[ref_id][sent_id])
                for aspect in data['human_score'].keys():
                    flatten_data['human_score'][aspect].append([human_score[aspect][sent_id][ann_id]])
                if len(unused_model_ids) == 0:  # all systems are evaluated in human evaluation.
                    # Just append each sentence.
                    for sys_id in range(num_systems):
                        flatten_data['hypotheses'][sys_id].append(hypotheses[sys_id][sent_id])
                    continue
                # Below, we want to replace unused model outputs with used model outputs.
                # Any used model's outputs is okay, so we choose the first system's output.
                used_hyp = hypotheses[used_model_ids[0]][sent_id]
                for sys_id in unused_model_ids:
                    # Replace.
                    hypotheses[sys_id][sent_id] = used_hyp
                # Now, the hypotheses only contains the sentences that have been evaluated by human (max five sentences).
                #   This ensures that LLMKobayashi24** metrics uses the human evaluated sentences.
                assert len(list(set(hypotheses[sys_id][sent_id] for sys_id in range(num_systems)))) <= 5, f"{sent_id=}\n{used_model_ids=}\n{human_score[sent_id][0]=}\n{len(list(set(hypotheses[sys_id][sent_id] for sys_id in range(num_systems))))}"
                # Expands each sentence.
                for sys_id in range(num_systems):
                    flatten_data['hypotheses'][sys_id].append(hypotheses[sys_id][sent_id])
                

        num_srcs = len(flatten_data['sources'])
        for hyp in flatten_data['hypotheses']:
            assert num_srcs == len(hyp), f"{num_srcs=}, {len(hyp)=}"
        for ref in flatten_data['references']:
            assert num_srcs == len(ref), f"{num_srcs=}, {len(ref)=}"
        for h_score in flatten_data['human_score'].values():
            assert num_srcs == len(h_score), f"{num_srcs=}, {len(h_score)=}"
        return flatten_data
    

    @abc.abstractmethod
    def corr_sentence(self, metric: MetricBase):
        '''Compute sentence-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            **SentenceCorrOutput: The sentence-level correlations output.
        '''
        orig_data = self.sentence_data
        if isinstance(metric, LLMKobayashi24):
            data = self.rearange_sent_data(orig_data)
        else:
            data = orig_data
        pairwise_score = metric.score_pairwise(
            **inputs_handler(
                metric, data['sources'], data['hypotheses'], data['references']
            ),
        )  # (num_sentence, num_systems, num_systems)
        corrs = []
        num_sents = len(data['sources'])
        num_sys = len(data['models'])
        for name in sorted(list(data['human_score'].keys())):
            human_scores = data['human_score'][name]
            agree = 0
            not_agree = 0
            denominator = 0
            none_count = 0

            for src_id in range(num_sents):
                for annotate_id in range(len(human_scores[src_id])):
                    for sys1, sys2 in itertools.combinations(range(num_sys), 2):
                        # The human score is minus ranking value,
                        #   so higher values indicate higher quality.
                        h1 = human_scores[src_id][annotate_id][sys1]
                        h2 = human_scores[src_id][annotate_id][sys2]
                        if None in [h1, h2]:
                            continue
                        if h1 == h2:
                            continue
                        denominator += 1
                        human_judge = 1 if h1 > h2 else -1
                        # SEEDA considers metric's tie result a loss.
                        metric_judge = pairwise_score[src_id][sys1][sys2]
                        if metric_judge == 0:
                            metric_judge = -1
                        if metric_judge == human_judge:
                            agree += 1
                        else:
                            if metric_judge is None:
                                none_count += 1
                            not_agree += 1
                            
            corr = self.Corr()
            corr.accuracy = agree / denominator
            corr.kendall = (agree - not_agree) / denominator
            corrs.append(corr)
        return corrs
    
    def window_analysis_system(
        self,
        metric: MetricBase,
        window: int=4,
        aggregation='default'
    ) -> "SEEDAWindowAnalysisSystemCorrOutput":
        '''System-level window analysis.

        Args:
            metric (MetricBase): The metric to be evaluated.
            window (int): The window size.

        Returns:
            SEEDAWindowAnalysisSystemCorrOutput: The correlations.
                - Contains .ew_edit, .ew_sent, .ts_edit, .ts_sent.
                - Each is a dictinary: {(start_rank, end_rank): Corr}.
        '''
        data = self.system_data
        system_results = self.corr_system(
            metric, aggregation=aggregation
        )
        corrs = []
        num_systems = len(data['models'])
        for name in self.SCORE_ID:
            raw_h_score = system_results.__dict__[name.lower()].human_scores
            metric_scores = system_results.__dict__[name.lower()].metric_scores
            # Sort both metric's and human's scores by the human score
            scores = sorted(
                list(zip(metric_scores, raw_h_score)),
                key=lambda x: x[1], reverse=True)
            m_score = [s[0] for s in scores]
            h_score = [s[1] for s in scores]
            corr = [
                self.Corr(
                    pearson=float(pearsonr(
                        m_score[i:i+window],
                        h_score[i:i+window]
                    )[0]),
                    spearman=float(spearmanr(
                        m_score[i:i+window],
                        h_score[i:i+window]
                    )[0])
                ) for i in range(num_systems-window+1)
            ]
            corrs.append({(i, i+window-1): corr[i] for i in range(num_systems-window+1)})
        return corrs

    def window_analysis_plot(
        self,
        results: dict[tuple, Corr]
    ):
        keys = sorted(list(results.keys()))
        pea = [results[k].pearson for k in keys]
        spe = [results[k].spearman for k in keys]
        x = list(range(len(pea)))
        fig, ax = plt.subplots()
        ax.plot(x, pea, label='Pearson')
        ax.plot(x, spe, label='Spearman')
        ax.legend()
        ax.grid(alpha=0.5)
        ax.set_xticks(x, [xx+1 for xx in x])
        return fig
            
    def pairwise_analysis(
        self,
        metric: MetricBase
    ):
        '''Compute sentence-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            **SentenceCorrOutput: The sentence-level correlations output.
        '''
        data = self.sentence_data
        pairwise_score = metric.score_pairwise(
            **inputs_handler(
                metric, data['sources'], data['hypotheses'], data['references']
            ),
        )  # (num_sentence, num_systems, num_systems)
        num_sents = len(data['sources'])
        num_sys = len(data['models'])
        stats = dict()
        for name in sorted(list(data['human_score'].keys())):
            human_scores = data['human_score'][name]
            denominator = 0
            stats[name] = stats.get(name, dict())
            for src_id in range(num_sents):
                for annotate_id in range(len(human_scores[src_id])):
                    for sys1, sys2 in itertools.combinations(range(num_sys), 2):
                        # The human score is minus ranking value,
                        #   so higher values indicate higher quality.
                        h1 = human_scores[src_id][annotate_id][sys1]
                        h2 = human_scores[src_id][annotate_id][sys2]
                        if None in [h1, h2]:
                            continue
                        if h1 == h2:
                            continue
                        denominator += 1
                        human_judge = 1 if h1 > h2 else -1
                        key = tuple(sorted([h1, h2]))
                        stats[name][key] = stats[name].get(key, {'agree': 0, 'not-agree': 0})
                        # SEEDA considers metric's tie result a loss.
                        metric_judge = pairwise_score[src_id][sys1][sys2]
                        if metric_judge == 0:
                            metric_judge = -1
                        if metric_judge == human_judge:
                            stats[name][key]['agree'] += 1
                        else:
                            stats[name][key]['not-agree'] += 1
            
            stats[name] = {
                k: stats[name][k]['agree'] / (stats[name][k]['agree'] + stats[name][k]['not-agree'])
                for k in stats[name]
            }
            # sort by the span index
            stats[name] = sorted(stats[name].items(), key=lambda x: x[1])
        return stats
    
    def pairwise_analysis_plot(
        self,
        results: list[tuple, float]
    ):
        plt.figure(figsize=(10, 8))
        x_vals = [-1 * pair[0][0] for pair in results]  # rank A
        y_vals = [-1 * pair[0][1] for pair in results]  # rank B
        z_vals = [pair[1] for pair in results]  # accuracy
        df = pd.DataFrame({"x": x_vals, "y": y_vals, "z": z_vals})
        heatmap_data = df.pivot(index="y", columns="x", values="z")
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="coolwarm",
            center=0,
            cbar=True,
            fmt=".2f",
            annot_kws={"size": 35, "weight": "bold"}
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=35, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=35, fontweight="bold")
        ax.set_xlabel("Rank B", fontsize=35, fontweight='bold')
        ax.set_ylabel("Rank A", fontsize=35, fontweight='bold')
        ax.xaxis.set_label_position('top') 
        ax.xaxis.tick_top()

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(labelsize=35)
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(35)
            label.set_fontweight("bold")
        return ax.get_figure()