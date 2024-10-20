import argparse

def main():
    args = get_parser()
    srcs_full = open(args.src_full).read().rstrip().split('\n')
    srcs_subset = open(args.src_subset).read().rstrip().split('\n')
    refs = open(args.ref).read().rstrip().split('\n')
    subset_indices = [i for i, s in enumerate(srcs_full) if s in srcs_subset]
    refs_subset = [refs[i] for i in subset_indices]
    with open(args.out, 'w') as f:
        f.write('\n'.join(refs_subset))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_full', required=True)
    parser.add_argument('--src_subset', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()