import argparse
from gecommon import Parallel

def main():
    args = get_parser()
    gec = Parallel.from_m2(args.m2, ref_id=args.ref_id)
    print('\n'.join(gec.trgs))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m2', required=True)
    parser.add_argument('--ref_id', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()