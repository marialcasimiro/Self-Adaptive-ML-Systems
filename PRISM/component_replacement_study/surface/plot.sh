#!/usr/bin/env bash

""" SNAPSHOT TIME == 0 """
snapshot_time=0
rbModelTPR=75


retrainCost=8
replaceCost=0
settings=snapshotTime_$snapshot_time-rbModelTPR_$rbModelTPR-replaceCost_$replaceCost-retrainCost_$retrainCost
file_1=../results/parsed_res-tactic_nop-$settings.txt
file_2=../results/parsed_res-tactic_retrain-$settings.txt
file_3=../results/parsed_res-tactic_replace-$settings.txt
output_file=./sel-$settings.tex
template_file=./templates/selgrid20x10.tex

python3.8 processsel.py  $file_1 $file_2 $file_3 $output_file $template_file    

pdflatex -output-directory=. $output_file



retrainCost=5
replaceCost=8
settings=snapshotTime_$snapshot_time-rbModelTPR_$rbModelTPR-replaceCost_$replaceCost-retrainCost_$retrainCost
file_1=../results/parsed_res-tactic_nop-$settings.txt
file_2=../results/parsed_res-tactic_retrain-$settings.txt
file_3=../results/parsed_res-tactic_replace-$settings.txt
output_file=./sel-$settings.tex
template_file=./templates/selgrid20x10.tex

python3.8 processsel.py  $file_1 $file_2 $file_3 $output_file $template_file    

pdflatex -output-directory=. $output_file



retrainCost=8
replaceCost=15
settings=snapshotTime_$snapshot_time-rbModelTPR_$rbModelTPR-replaceCost_$replaceCost-retrainCost_$retrainCost
file_1=../results/parsed_res-tactic_nop-$settings.txt
file_2=../results/parsed_res-tactic_retrain-$settings.txt
file_3=../results/parsed_res-tactic_replace-$settings.txt
output_file=./sel-$settings.tex
template_file=./templates/selgrid20x10.tex

python3.8 processsel.py  $file_1 $file_2 $file_3 $output_file $template_file    

pdflatex -output-directory=. $output_file

rm ./*.aux
rm ./*.log
rm ./*.gz