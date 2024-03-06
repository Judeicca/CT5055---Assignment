#import libaries
import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot(df: pd.DataFrame, x1: str, x2: str, y: str, title: str = '', save: bool = False, figname='figure.png'):
    plt.figure(figsize=(14, 7))
    
    
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

locations = ["Headquarters", "BranchA", "BranchB", "BranchC", "BranchD", "BranchE", "BranchF", "BranchG", "BranchH", "BranchI", "BranchJ", "BranchK", "BranchL", "BranchM", "BranchN", "BranchO", "BranchP", "BranchQ", "BranchR", "BranchS", "BranchT", "BranchU", "BranchV", "BranchW", "BranchX", "BranchY", "BranchZ"]

distance = 0.5

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=1)

df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)

