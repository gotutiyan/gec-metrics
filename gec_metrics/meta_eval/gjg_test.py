from gec_metrics.metrics import MetricBaseForReferenceFree
from .gjg import MetaEvalGJG
import math

class Length(MetricBaseForReferenceFree):
    '''A pseudo-reference-free metric with character count as the score.
    '''
    def score_corpus(
        self,
        sources,
        hypotheses,
    ):
        return sum(len(h) for h in hypotheses)
    
    def score_sentence(
        self,
        sources,
        hypotheses,
    ):
        return [len(h) for h in hypotheses]
    
def test_corr_system():
    scorer = Length(Length.Config())
    meta_seeda = MetaEvalGJG(MetaEvalGJG.Config())
    out = meta_seeda.corr_system(scorer)
    
    corr_cls = MetaEvalGJG.Corr
    gold = MetaEvalGJG.GJGSystemCorrOutput(
        ew=corr_cls(pearson=-0.175093788793105, spearman=-0.016483516483516484),
        ts=corr_cls(pearson=-0.2232996476764572, spearman=-0.06043956043956044),
    )
    for k in gold.__dict__:
        assert math.isclose(gold.__dict__[k].pearson, out.__dict__[k].pearson, rel_tol=1e-6)
        assert math.isclose(gold.__dict__[k].spearman, out.__dict__[k].spearman, rel_tol=1e-6)