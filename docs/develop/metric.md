# Metric Development 

When developing a new metric, it is recommended to inherit from one of the `MetricBase**` classes. This ensures that the interface is consistent with existing metrics, making it easier to perform tasks such as meta-evaluation.

Several `MetricBase**` classes are available, primarily designed for different evaluation types. Choose one according to the metric you intend to implement:
- `MetricBaseForReferenceBased`: Class for reference-based metrics.
- `MetricBaseForReferenceFree`: Class for reference-free metrics.
- `MetricBaseForSourceFree`: Class for metrics that evaluate using only hypotheses and references (e.g., BERTScore).

After implementing a metric by inheriting from one of these base classes, you can perform meta-evaluation using the meta-evaluation components of `gec-metrics`.

## An exmaple

As an example, we will develop a toy metric that measures the difference in word count between the source sentence and the hypothesis sentence. Since it can be considered a reference-free evaluation metric, we implement it by inheriting from `MetricBaseForReferenceFree`.

```python
from gec_metrics.metrics import MetricBaseForReferenceFree

class LengthDiff(MetricBaseForReferenceFree):
    def score_sentence(self, sources, hypotheses):
        return [abs(len(s.split(' ')) - len(h.split(' '))) for s, h in zip(sources, hypotheses)]
```

Once implemented, create an instance and perform meta-evaluation using the `corr_system()` method of a meta-evaluation class. (Of course, since it is a toy metric, the resulting correlations will also be meaningless.)

```python
from gec_metrics.meta_eval import MetaEvalSEEDA
metric = LengthDiff()
meta = MetaEvalSEEDA()
results = meta.corr_system(metric)
print(results)
```