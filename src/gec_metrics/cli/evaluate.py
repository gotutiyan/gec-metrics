import gec_metrics
from gec_metrics.metrics import (
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree
)
import argparse
import yaml

def read_lines(path):
    sents = open(path).read().rstrip().split('\n')
    return [s.strip() for s in sents]

def read_yaml(path):
    with open(path, 'r') as yml:
        config = yaml.safe_load(yml)
    return config

def main():
    args = get_parser()
    metric_cls = gec_metrics.get_metric(args.metric)
    if args.config is not None:
        metric_config = read_yaml(args.config)[args.metric]
    else:
        metric_config = {}
    scorer = metric_cls(metric_cls.Config(**metric_config))
    srcs = read_lines(args.src)
    for hyp in args.hyps:
        hyps = read_lines(hyp)
        if isinstance(scorer, MetricBaseForReferenceBased):
            assert args.refs is not None
            refs = [read_lines(r) for r in args.refs]
            score = scorer.score_corpus(
                srcs, hyps, refs
            )
        elif isinstance(scorer, MetricBaseForReferenceFree):
            score = scorer.score_corpus(
                srcs, hyps
            )
        print(f'Score={score} | Metric={args.metric} | hyp_file={hyp}')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Sources file.')
    parser.add_argument('--hyps', nargs='+', required=True, help='Hypotheses files.')
    parser.add_argument('--refs', nargs='+', help='References files.')
    parser.add_argument('--metric', required=True, choices=gec_metrics.get_metric_ids(), help='ID of the metric.')
    parser.add_argument('--config', help='YAML-based config file.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
