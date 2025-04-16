# Metric Introduction

## Import

Each metric is implemented as a class. You can import the class from `gec_metrics.metrics.`:

```python
from gec_metrics.metrics import ERRANT
```

Alternatively, you can use the `get_metric()` method. A list of available metric IDs can be obtained using `get_metric_ids()`.

```python
from gec_metrics import get_metric, get_metric_ids
metric_cls = get_metric('errant')
print(get_metric_ids())  # ['scribendi', 'impara', 'some', ...
```

## Initialize

A Config class is defined for each class and can be passed during initialization. For example, the beta parameter for the ERRANT F-beta metric can be specified via Config. If no config is provided, the default settings are used.

```python
from gec_metrics.metrics import ERRANT
metric = ERRANT(ERRANT.Config(
    beta=0.5
))

# When using the default configuration.
metric = ERRANT()
```

You can see the default configuration via `<metric_class>.Config()`:

```python
from gec_metrics.metrics import ERRANT
print(ERRANT.Config())
# ERRANT.Config(beta=0.5, language='en')
```

## Evaluate

The metric classes support various types of evaluation via a unified interface.

### Single system evaluation
When evaluating a single system, `score_corpus()` can be used for corpus-level evaluation.

```python
from gec_metrics.metrics import ERRANT
sources = ['A sample sentnce .', 'Another a sample sentence .']
hypotheses = ['A sample sentence .', 'Another a sample sentence .']
# Assume that the reference shape is (num_references, num_sentences)
references = [['A sample sentence .', 'Another sample sentence .']]

metric = ERRANT()
score = metric.score_corpus(sources, hypotheses, references)
print(score)  # 0.8333
```

Some metrics have the verbose version to obtain detailed results.
```python
score = metric.score_corpus_verbose(sources, hypotheses, references)
print(score)
'''F-0.5=0.8333333333333334
 Prec=1.0
 Rec=0.5
 TP=1.0, FP=0.0, FN=1.0, TN=0.0
'''
```

You can also use `score_sentence()` to get sentence-level scores.

```python
scores = metric.score_sentence(sources, hypotheses, references)
print(scores)  # [1.0, 0.0]
```

### Multiple system evaluation
To rank the outputs of multiple systems, `rank_systems()` can be used. Since there are several variations in the ranking method, it can be specified using the aggregation= parameter. 
- The `aggregation="default"` method simply calculates the corpus-level evaluation score for each system.
- When `aggregation="trueskill"` or `aggregation="expected_wins"` is specified, the system-level scores are directly derived from sentence-level scores using a rating algorithm.

```python
hyps1 = gec_model1(sources)
hyps2 = gec_model2(sources)
hyps3 = gec_model3(sources)
scores: list[float] = metric.rank_systems(
    sources,
    [hyps1, hyps2, hyps3],
    references,
    aggregation="default"
)
print(scores) 
```

`score_pairwise()` takes the outputs of multiple systems as input and calculates sentence-level pairwise evaluation scores. Although it may not often be used directly, it is implemented to facilitate the calculations within rank_systems().

The output shape is (num_sents, num_systems, num_systems). Each element can be accessed using `[sent_id][sys_id1][sys_id2]` and takes a value of -1, 0, or 1. A value of -1 indicates that sys_id1 loses to sys_id2, 0 indicates a tie, and 1 indicates that sys_id1 wins against sys_id2

```python
hyps1 = gec_model1(sources)
hyps2 = gec_model2(sources)
hyps3 = gec_model3(sources)
pair_scores: list[list[list[float]]] = metric.rank_pairwise(
    sources,
    [hyps1, hyps2, hyps3],
    references
)
print(pair_scores)
'''Output example.
[
    [
        [0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0]
    ],
    [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ]
]
'''
```
