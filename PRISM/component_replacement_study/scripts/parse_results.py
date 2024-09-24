#!/usr/bin/env python
import os
import sys
import subprocess as sub


def execBashCMD(cmd):

    sub.call(cmd, shell = True)


def parse(ff, file_name, output_file):
    # adv-snapshotTime_271-rbModelTPR_50-rbModelTNR_50-replaceCost_0-retrainCost_1-tactic_nop
    tokens = file_name.split("-")
    rbModelTPR = tokens[2].split("_")[1]
    rbModelTNR = tokens[3].split("_")[1]
    if len(tokens) >= 9: 
        deltaTNRretrain = -int(tokens[7])
    else:
        deltaTNRretrain = int(tokens[6].split("_")[1])

    initTNRretrain = 98
    newTNRretrain = initTNRretrain + deltaTNRretrain

    lines = ff.readlines()
    for line in lines:
        if "Result" in line:
            continue
        sysU = line

    new_str = f"{newTNRretrain}\t{rbModelTNR}\t{sysU}"
    output_file.write(new_str)


def main():

    if len(sys.argv) <= 1:
        print("Usage: %s RESULTS_DIR" % sys.argv[0])
        sys.exit(0)

    print("-------- MERGING PRISM RESULTS --------")

    results_dir = sys.argv[1]

    for root, subdirs, _ in os.walk(results_dir):
        for subdir in subdirs:
            output_file = open(results_dir + 'parsed_res.txt', "w")
            # output_file.write("rbModelTPR\trbModelTNR\tsysU\n")
            output_file.write("newTNRretrain\trbModelTNR\tsysU\n")

            snapshot_time = ""
            for f in os.listdir(results_dir + subdir):
                if f.endswith('.txt') and "adv" in f and "snapshotTime_0" in f and "replaceCost_15-" in f and "retrainCost_8-" in f and "newTNRretrain_" in f:
                    # adv-snapshotTime_0-rbModelTPR_75-rbModelTNR_98-replaceCost_0-retrainCost_8-newTNRretrain_-1-tactic_replace.tra
                    print(f"file: {subdir}/{f}")
                    snapshot_time = f.split("-")[1]
                    rbModelTPR = f.split("-")[2]
                    rbModelTNR = f.split("-")[3]
                    replaceCost = f.split("-")[4]
                    retrainCost = f.split("-")[5]
                    tactic = f.split("-")[7][:-4]

                    ff = open(f"{results_dir}{subdir}/{f}", "r")
                    parse(ff, f, output_file)
                    ff.close()

            output_file.close()

            """ sort results file 
                -n : sort in numerical order
                -k : sort first by field 1 and then by field 2
            """
            execBashCMD(f"sort -n -k 1 -k 2 {results_dir}parsed_res.txt -o {results_dir}parsed_res.txt")

            """ rename results file """
            execBashCMD(f"mv {results_dir}parsed_res.txt {results_dir}parsed_res-tactic_{subdir}-{snapshot_time}-{rbModelTPR}-{replaceCost}-{retrainCost}.txt ")

if __name__ == '__main__':
    main()