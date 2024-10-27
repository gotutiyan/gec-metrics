# gec-metrics
A library for evaluation of Grammatical Error Correction

# Install
```sh
pip install git+https://github.com/gotutiyan/gec-metrics
python -m spacy download en
```
Or,
```sh
git clone git@github.com:gotutiyan/gec-metrics.git
cd gec-metrics
pip install -e ./
python -m spacy download en
```

# Common Usage
All metrics has the same interface.  
This is an GLEU exmaple.
```python
from gec_metrics.metrics import GLEU
scorer = GLEU(GLEU.Config())
srcs = ['This sentences contain grammatical error .']
hyps = ['This sentence contains an grammatical error .']
refs = [
    ['This sentence contains an grammatical error .'],
    ['This sentence contains grammatical errors .']
]
# Corpus-level score
# If the metric is reference-free, the argument `references=` is not needed.
corpus_score: float = scorer.score_corpus(
    sources=srcs,
    hypotheses=hyps,
    references=refs
)
# Sentence-level scores
sent_scores: list[float] = scorer.score_sentence(
    sources=srcs,
    hypotheses=hyps,
    references=refs
)
```

# Metrics

We introduce which metrics are available.  
All of arguments in the following examples indicate default values.

## Reference-based
### M2 

To be added.

###  GLEU+ [[Napoles+ 15]](https://aclanthology.org/P15-2097/) [[Napoles+ 16]](https://arxiv.org/abs/1605.02592)  

```python
from gec_metrics.metrics import GLEU
scorer = GLEU(GLEU.Config(
    iter=500,  # The number of iterations 
    n=4  # max n-gram
))
```
We also provide a reproduction of the official implementation as GLEUOfficial.  
The official one ignores ngram frequency differences when calculating the difference set between source and reference.
```python
from gec_metrics.metrics import GLEUOfficial
scorer = GLEUOfficial(GLEUOfficial.Config(
    iter=500,  # The number of iterations 
    n=4  # max n-gram
))
```

### ERRANT [[Felice+ 16]](https://aclanthology.org/C16-1079/) [[Bryant+ 17]](https://aclanthology.org/P17-1074/)
```python
from gec_metrics.metrics import ERRANT
scorer = ERRANT(ERRANT.Config(
    beta=0.5,  # The beta for F-beta score
    language='en'  # Language for SpaCy.
))
```

### GoToScorer [[Gotou+ 20]](https://aclanthology.org/2020.coling-main.188/)

```python
from gec_metrics.metrics import GoToScorer
scorer = GoToScorer(GoToScorer.Config(
    beta=0.5,  # The beta for F-beta score
    ref_id=0,  # The reference id
    no_weight=True,  # If True, all weights are 1.0
    weight_file=None  # It is required if no_weight=False
))
```
You need to generate a weight file via `gen-weight-for-gotoscorer`.  
The output is a JSON file.  
```sh
gen-weight-for-gotoscorer \
    --src <raw text file> \
    --ref <raw text file> \
    --hyp <raw text file 1> <raw text file 2> ... <raw text file N> \
    --out weight.json
```

### PT-M2 [[Gong+ 22]](https://aclanthology.org/2022.emnlp-main.463/)

To be added.

### PT-ERRANT [[Gong+ 22]](https://aclanthology.org/2022.emnlp-main.463/)

To be added.

### CLEME [[Ye+ 23]](https://aclanthology.org/2023.emnlp-main.378/)

To be added.

### GREEN [[Koyama+ 24]](https://aclanthology.org/2024.inlg-main.25/)
```python
from gec_metrics.metrics import GREEN
scorer = GREEN(GREEN.Config(
    n=4,  # Max n of ngram
    beta=2.0,  # The beta for F-beta
    unit='word'  # 'word' or 'char'. Choose word-level or character-level
))
```

## Reference-based (without sources)

These metrics are intended to be used for a component of PT-{M2, ERRANT}, but are also exposed to API.

### BERTScore [[Zhang+ 19]](https://arxiv.org/abs/1904.09675)
To be added.

### BARTScore [[Yuan+ 21]](https://proceedings.neurips.cc/paper/2021/hash/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Abstract.html)

To be added.

## Reference-free

### SOME [[Yoshimura+ 20]](https://aclanthology.org/2020.coling-main.573/)  
Download pre-trained models in advance from [here](https://github.com/kokeman/SOME#:~:text=Download%20trained%20model).
```python
from gec_metrics.metrics import SOME
scorer = SOME(SOME.Config(
    model_g='gfm-models/grammer',
    model_f='gfm-models/fluency',
    model_m='gfm-models/meaning',
    weight_f=0.55,
    weight_g=0.43,
    weight_m=0.02,
    batch_size=32
))
```
### Scribendi [[Islam+ 21]](https://aclanthology.org/2021.emnlp-main.239/)
```python
from gec_metrics.metrics import Scribendi
scorer = Scribendi(Scribendi.Config(
    model='gpt2',  # The model name or path to the language model to compute perplexity
    threshold=0.8  # The threshold for the maximum values of token-sort-ratio and levelshtein distance ratio
))
```
### IMPARA [[Maeda+ 22]](https://aclanthology.org/2022.coling-1.316/)  
Note that the QE model is an unofficial model which achieves comparable correlation with the human evaluation results.  
By default, it uses an unofficial pretrained QE model: [[gotutiyan/IMPARA-QE]](https://huggingface.co/gotutiyan/IMPARA-QE).
```python
from gec_metrics.metrics import IMPARA
scorer = IMPARA(IMPARA.Config(
    model_qe='gotutiyan/IMPARA-QE',  # The model name or path for quality estimation.
    model_se='bert-base-cased',  # The model name or path for similarity estimation.
    threshold=0.9  # The threshold for the similarity score.
))
```

# Meta Evaluation
To perform meta evaluation easily, we provide meta-evaluation scripts.

### Preparation
To donwload test data and human scores, you must download datasets by using the shell.
```sh
bash prepare_meta_eval.sh
```

This scripts creates `meta_eval_data/` directory which consists of SEEDA dataset and CoNLL14 official submissions.
```
meta_eval_data/
├── GJG15
│   └── judgments.xml
├── conll14
│   ├── official_submissions
│   │   ├── AMU
│   │   ├── CAMB
│   │   ├── ...
│   ├── REF0
│   └── REF1
└── SEEDA
    ├── outputs
    │   ├── all
    │   │   ├── ...
    │   └── subset
    │       ├── ...
    ├── scores
    │   ├── human
    │   │   ├── ...├── ...
```

### SEEDA: [[Kobayashi+ 24]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00676/123651/Revisiting-Meta-evaluation-for-Grammatical-Error)
The examples below uses ERRANT as a metric, but can also use other metrics based on `gec_metrics.metrics.MetricBase`.  
- `ew_` means using ExpectedWins human evaluation scores and `ts_` means using TrueSkill.

```python
from gec_metrics.metrics import MetricBase, GLEU
from gec_metrics.meta_eval import MetaEvalSEEDA
scorer = GLEU(GLEU.Config())
meta_seeda = MetaEvalSEEDA(
    MetaEvalSEEDA.Config(system='base')
)
# System correlation
results = meta_seeda.corr_system(scorer)
# Output:
# SEEDASystemCorrOutput(ew_edit=Corr(pearson=0.9007842791853424,
#                                    spearman=0.9300699300699302,
#                                    accuracy=None,
#                                    kendall=None),
#                       ew_sent=Corr(pearson=0.8749437873537543,
#                                    spearman=0.9090909090909092,
#                                    accuracy=None,
#                                    kendall=None),
#                       ts_edit=Corr(pearson=0.9123732084071973,
#                                    spearman=0.9440559440559443,
#                                    accuracy=None,
#                                    kendall=None),
#                       ts_sent=Corr(pearson=0.8856173179230024,
#                                    spearman=0.9020979020979022,
#                                    accuracy=None,
#                                    kendall=None))

# Sentence correlation
results = meta_seeda.corr_sentence(scorer)
# Output:
# SEEDASentenceCorrOutput(sent=Corr(pearson=None,
#                                   spearman=None,
#                                   accuracy=0.6715701950751519,
#                                   kendall=0.3431403901503038),
#                         edit=Corr(pearson=None,
#                                   spearman=None,
#                                   accuracy=0.6734561494551116,
#                                   kendall=0.3469122989102231))
```

### GJG15: [[Grundkiewicz+ 15]](https://aclanthology.org/D15-1052/)

This is referred to `GJG15` in the SEEDA paper.  
Basically, TrueSkill ranking is used to compute the correlation.

```python
from gec_metrics.metrics import MetricBase, GLEU
from gec_metrics.meta_eval import MetaEvalGJG
scorer = GLEU(GLEU.Config())
meta_gjg = MetaEvalGJG(MetaEvalGJG.Config())
# System correlation
results = meta_gjg.corr_system(scorer)
# Output:
# GJGSystemCorrOutput(ew=Corr(pearson=0.601290729078602,
#                             spearman=0.5934065934065934,
#                             accuracy=None,
#                             kendall=None),
#                     ts=Corr(pearson=0.6633835644883472,
#                             spearman=0.6868131868131868,
#                             accuracy=None,
#                             kendall=None))

results = meta_gjg.corr_sentence(scorer)
# Output:
# GJGSentenceCorrOutput(corr=Corr(pearson=None,
#                                 spearman=None,
#                                 accuracy=0.6729157079690282,
#                                 kendall=0.34583141593805644))
```

# Correlations can be obtained

Using scripts for the metrics and the meta-evaluation in this repository, we performed meta-evaluation.  
The table below is the result.

