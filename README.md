# COMPSCI-X433.3

Final project analyzing all loans issued from 2007 to 2015 via LendingClub, the world’s largest peer-to-peer lending company.

Project members: 
Zhicheng (Jason) Xue
Fiona Tang

The dataset can be found at https://www.kaggle.com/wendykan/lending-club-loan-data/data.

We used Anaconda Spyder environment and Jupyter notebook to execute. We used Python version 3.6.

The code relies on numpy, scipy, pandas, matplotlib, plotly, seaborn, tensorflow, and sklearn. The specific modules are below (and are also imported into memory upon execution):
import numpy as np
from scipy.stats import normaltest, anderson, shapiro
import pandas as pd
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode()
import seaborn as sns
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from tensorflow.contrib.factorization import KMeans
