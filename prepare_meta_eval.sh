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
m2-to-raw --m2 conll14st-test-data/noalt/official-2014.combined.m2 --ref_id 0 > $CONLL14DIR/REF0
m2-to-raw --m2 conll14st-test-data/noalt/official-2014.combined.m2 --ref_id 1 > $CONLL14DIR/REF1

for i in 0 1; do
make-subset-ref \
    --src_full $DIR/SEEDA/outputs/all/INPUT.txt \
    --src_subset $DIR/SEEDA/outputs/subset/INPUT.txt \
    --ref $DIR/conll14/REF$i \
    --out $DIR/SEEDA/outputs/subset/REF$i.txt
done

make-subset-ref \
    --src_full $DIR/SEEDA/outputs/all/INPUT.txt \
    --src_subset $DIR/SEEDA/outputs/subset/INPUT.txt \
    --ref $DIR/SEEDA/outputs/all/LM-Critic.txt \
    --out $DIR/SEEDA/outputs/subset/LM-Critic.txt

rm official_submissions.tar.gz
rm conll14st-test-data.tar.gz
rm -r conll14st-test-data

mkdir $DIR/GJG15/
cd $DIR/GJG15/
wget https://github.com/grammatical/evaluation/raw/refs/heads/master/data/judgments.xml
