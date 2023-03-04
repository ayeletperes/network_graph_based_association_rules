# imports to deploy the main
import argparse
# data processing
import pandas as pd
import numpy as np
from pathlib import Path
import glob
# machine learning
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
# pattern mining
from efficient_apriori import apriori # add to requirements
from mlxtend.preprocessing import TransactionEncoder # add to requirements
from mlxtend.frequent_patterns import fpgrowth # add to requirements
from mlxtend.frequent_patterns import association_rules # add to requirements
# network graphs
import networkx as nx # add to requirements
from tqdm.auto import tqdm # add to requirements
from multiprocessing import Pool
# import from modules
from data_processing import genotypeData
from rules_creation import rulesMining
from network_model import rulesNetwork

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str,
                        help="a path to the directory of the genotype tables")
    parser.add_argument("--file_pattern", type=str, default= "_genotype",
                        help="The pattern to catch the genotype files in the directory")
    parser.add_argument("--metadata", type=str, help="A tab delimited metadata file with the subject and status columns.")
    parser.add_argument("--mining_algorithem", type=str, default="fpgrowth", help="Which rule mining algorithem to use. Either fpgrowth from mlxtend or apriori from efficient_apriori.")
    parser.add_argument("--greedy_path_metric", type=str, default="increase_factor", help="which metric to use for the gready search: increase_factor - the increase value;confidence - the rule confidence value;lift - the rule lift value;lif - the increase factor divided by the lift value;cif - the increase factor multiplied by the confidence")
    parser.add_argument("--path_steps", type=int, default = 10, help="The maximum nuber of steps to take in the greedy path search")
    parser.add_argument("--processes", type=int, default = 2, help="How many processes to use for the network creation")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.data_folder
    file_pattern = args.file_pattern
    algo = args.mining_algorithem
    metric = args.greedy_path_metric
    path_steps = args.path_steps
    processes = args.processes
    metadata = pd.read_csv(args.metadata, delimiter= "\t")
    if {'subject','status'}.issubset(set(metadata.columns)):
        d_metadata = metadata.set_index('subject')['status'].T.to_dict()
    else:
        raise ValueError(" ".join(['subject','status'])+" are required in the metadata table")

    gd = genotypeData(path, d_metadata, file_pattern)

    # get the files
    files = gd.list_files()
    # read the genotype data
    genotypes = gd.read_files(files)
    # clean the data
    genotypes_clean = gd.clean_genotype_data(genotypes, silent=True)

    # mining rules
    mr = rulesMining(genotypes_clean, transactions_columns="calls")
    rules = mr.mining_rules(algo=algo)

    # creating network
    net = rulesNetwork(rules, genotypes_clean)
    graph = net.create_network_multiprocess(processes=processes)
    path,steps,path_increase_value = net.greedy_max_weight(metric = metric, steps = path_steps)
    pred = net.get_prediction(path)
    cm = net.get_confusion_matrix(path)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    print('For the chosen path: {}'.format(list(set(path))))
    print('The F1 score prediction is {}'.format(pred))
    print('The confusion matrix is:')
    disp.plot()

if __name__ == '__main__':
    main()