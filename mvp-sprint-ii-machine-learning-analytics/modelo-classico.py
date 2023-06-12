# configuração para não exibir os warnings
import warnings
warnings.filterwarnings("ignore")

# imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from urllib.parse import quote
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# -----
# carga do dataset
uci_url = 'https://archive.ics.uci.edu/static/public/697/'
predict_student_uci_url = 'predict+students+dropout+and+academic+success.zip'
request = urllib.request.urlopen(uci_url + urllib.parse.quote(predict_student_uci_url))
zipfile = ZipFile(BytesIO(request.read()))
filepath = TextIOWrapper(zipfile.open('data.csv'), encoding='utf-8')
dataset = pd.read_csv(filepath, sep=';')

print(dataset)

# -----
# preparação dos dados

# separação em bases de treino e teste (holdout)
array = dataset.values
X = array[:,0:36]
Y = array[:,36]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

# criando os folds para a validação cruzada
num_splits = 10
kfold = KFold(n_splits=num_splits, shuffle=True, random_state=7)

print(X)
print(Y)


# -----
# modelagem

# definindo uma seed global para esta célula de código
np.random.seed(7)