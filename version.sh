#!/bin/sh
echo $(date) > version
echo 'Last commit' >> version
echo $(git rev-parse HEAD) >> version
echo '' >> version
echo 'Current branch' >> version
echo $(git branch --show-current) >> version

tar --exclude='*/__pycache__' -cvzf ./src.tar configs datasets.py discriminators generators inference.py submitSlurmjob.py train.py utils.py fid_evaluation.py metric_utils.py version
mkdir src
cd src
tar -zxvf ../src.tar
mv ../src.tar ./
rm ../version