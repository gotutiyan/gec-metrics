import argparse
from gec_metrics.metrics import GoToScorer
import json

def annotate_weight(
    source: str,
    reference: str,
    hypotheses: list[str],
    gotoscorer: GoToScorer
):
    ref_edits = gotoscorer.edit_extraction(
        src=source,
        trg=reference
    )
    hyp_edits_list = [
        gotoscorer.edit_extraction(
            src=source,
            trg=hyp
        ) for hyp in hypotheses
    ]
    ref_chunks = gotoscorer.generate_chunks(ref_edits, tokens=source.split(' '))
    evaluations = [[] for _ in ref_chunks]  # The shape will be (num_chunks, num_hyps)
    for hyp_edits in hyp_edits_list:
        hyp_chunks = gotoscorer.generate_chunks(hyp_edits, tokens=source.split(' '))
        for chunk_id, r_chunk in enumerate(ref_chunks):
            is_correct = False
            for h_chunk in hyp_chunks:
                if (r_chunk.o_start, r_chunk.o_end) \
                    == (r_chunk.o_start, r_chunk.o_end) \
                    and r_chunk.c_str == h_chunk.c_str:
                        is_correct = True
            
            evaluations[chunk_id].append(int(is_correct))
    weights = [1 - sum(e) / len(e) for e in evaluations]
    return {
        'evaluations': evaluations,
        'weights': weights
    }

def main():
    args = get_parser()
    gotoscorer = GoToScorer(GoToScorer.Config(no_weight=True))
    sources = open(args.src).read().rstrip().split('\n')
    references = open(args.ref).read().rstrip().split('\n')
    hypotheses = [open(h).read().rstrip().split('\n') for h in args.hyps]
    data = []
    for sent_id in range(len(sources)):
        results = annotate_weight(
            source=sources[sent_id],
            reference=references[sent_id],
            hypotheses=[hyp[sent_id] for hyp in hypotheses],
            gotoscorer=gotoscorer
        )
        data.append(results)
    with open(args.out, 'w') as fp:
        json.dump(data, fp, indent=2)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--hyps', nargs='+', required=True)
    parser.add_argument('--out', default='sample_weight.json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # args = get_parser()
    main()