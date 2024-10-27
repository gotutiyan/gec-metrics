from gec_metrics.metrics import MetricBaseForReferenceFree
from .seeda import MetaEvalSEEDA
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
    meta_seeda = MetaEvalSEEDA(MetaEvalSEEDA.Config())
    out = meta_seeda.corr_system(scorer)
    
    corr_cls = MetaEvalSEEDA.Corr
    gold = MetaEvalSEEDA.SEEDASystemCorrOutput(
        ew_edit=corr_cls(pearson=0.44776143599850254, spearman=0.4798606308117672),
        ew_sent=corr_cls(pearson=0.4889329068258237, spearman=0.5078816895453011),
        ts_edit=corr_cls(pearson=0.5001699042776178, spearman=0.6444843508712785),
        ts_sent=corr_cls(pearson=0.5236585407482779, spearman=0.5148869542286845)
    )
    for k in gold.__dict__:
        assert math.isclose(gold.__dict__[k].pearson, out.__dict__[k].pearson, rel_tol=1e-6)
        assert math.isclose(gold.__dict__[k].spearman, out.__dict__[k].spearman, rel_tol=1e-6)

def test_corr_sentence():
    scorer = Length(Length.Config())
    meta_seeda = MetaEvalSEEDA(MetaEvalSEEDA.Config())
    out = meta_seeda.corr_sentence(scorer)
    
    corr_cls = MetaEvalSEEDA.Corr
    gold = MetaEvalSEEDA.SEEDASentenceCorrOutput(
        edit=corr_cls(accuracy=0.5467047223663726, kendall=0.0934094447327452),
        sent=corr_cls(accuracy=0.5540987101588317, kendall=0.10819742031766336)
    )
    for k in gold.__dict__:
        assert math.isclose(gold.__dict__[k].accuracy, out.__dict__[k].accuracy, rel_tol=1e-6)
        assert math.isclose(gold.__dict__[k].kendall, out.__dict__[k].kendall, rel_tol=1e-6)
    