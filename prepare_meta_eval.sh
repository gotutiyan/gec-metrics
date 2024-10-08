#!/bin/bash
set -eu

DIR=meta_eval_data/
mkdir -p $DIR
git clone https://github.com/tmu-nlp/SEEDA.git $DIR/SEEDA

CONLL14DIR=$DIR/conll14
mkdir -p $CONLL14DIR

wget https://www.comp.nus.edu.sg/~nlp/conll14st/official_submissions.tar.gz
tar -xvf official_submissions.tar.gz
mv official_submissions $CONLL14DIR

wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xvf conll14st-test-data.tar.gz
cat conll14st-test-data/noalt/official-2014.combined.m2 | grep '^S' | cut -d ' ' -f 2- > $CONLL14DIR/official_submissions/INPUT
m2_to_raw --m2 conll14st-test-data/noalt/official-2014.combined.m2 --ref_id 0 > $CONLL14DIR/REF0
m2_to_raw --m2 conll14st-test-data/noalt/official-2014.combined.m2 --ref_id 1 > $CONLL14DIR/REF1

rm official_submissions.tar.gz
rm conll14st-test-data.tar.gz
rm -r conll14st-test-data


# Output: 
# meta_eval_data/
# ├── conll14
# │   ├── official_submissions
# │   │   ├── AMU
# │   │   ├── CAMB
# │   │   ├── CUUI
# │   │   ├── IITB
# │   │   ├── INPUT
# │   │   ├── IPN
# │   │   ├── NTHU
# │   │   ├── PKU
# │   │   ├── POST
# │   │   ├── RAC
# │   │   ├── SJTU
# │   │   ├── UFC
# │   │   └── UMC
# │   ├── REF0
# │   └── REF1
# └── SEEDA
#     ├── outputs
#     │   ├── all
#     │   │   ├── ...
#     │   └── subset
#     │       ├── ...
#     ├── scores
#     │   ├── human
#     │   │   ├── ...