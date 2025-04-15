# Meta-evaluation

gec-metrics also supports meta-evaluation. 

## Preparation
Please run the following command beforehand to download the relevant datasets:

```bash
gecmetrics-prepare-meta-eval
```

## Import

Each meta-evaluation benchmark is implemented as a single class. These classes can be imported from `gec_metrics.meta_eval`.


```python
from gec_metrics.meta_eval import MetaEvalSEEDA
```

Alternatively, you can use `get_meta_eval()`. Available IDs can be obtained using `get_meta_eval_ids()`.

```python
from gec_metrics import get_meta_eval, get_meta_eval_ids
metric_cls = get_metric('seeda')
print(get_metric_ids())  # [']
```

## Initialize

Similar to metrics, initialize them by passing a Config object as needed.

```python
from gec_metrics.meta_eval import MetaEvalSEEDA
meta = MetaEvalSEEDA(MetaEvalSEEDA.Config(
    system='base'
))

# When using the default configuration.
meta = MetaEvalSEEDA()
```

## Meta-evaluation

Meta-evaluation is often discussed at two levels: corpus-level and sentence-level, and gec-metrics supports both.

### System-Level Meta-Evaluation

System-level meta-evaluation is performed using `corr_system()`.
This function compares system scores obtained from human evaluation with the evaluation results from `metric.rank_systems()`. You can control how the metric scores are calculated by specifying the `aggregation=` argument.

The return value is a dedicated Output class. If the human evaluation includes multiple aspects, the correlations with all of them are calculated and returned.

```python
from gec_metrics.meta_eval import MetaEvalSEEDA
from gec_metrics.metrics import GREEN
metric = GREEN()
meta = MetaEvalSEEDA()
sys_results = meta.corr_system(metric, aggregation='default')
print(sys_results)
```

### Sentence-Level Meta-Evaluation

Sentence-level meta-evaluation is performed using `corr_sentence()`. The return value is a dedicated Output class. If the human evaluation includes multiple aspects, the correlations with all of them are calculated and returned.

```python
from gec_metrics.meta_eval import MetaEvalSEEDA
from gec_metrics.metrics import GREEN
metric = GREEN()
meta = MetaEvalSEEDA()
sent_results = meta.corr_sentence(metric, aggregation='default')
print(sent_results)
```