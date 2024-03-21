from sklearn import model_selection, metrics, preprocessing, datasets, ensemble, svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, f1_score, accuracy_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statistics
import plotly.express as px
import umap.umap_ as umap
import xgboost

df = pd.read_csv('data.csv')

def plot_cm (cm):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color = 'black')
    plt.show()
    
def mere_kvaliteta (test_y, pred_y):
    f1 = f1_score(test_y, pred_y, average='macro')
    auc = roc_auc_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y)
    print('F1 Score : %.2f' % f1)
    print('AUC ROC : %.2f' % auc)
    print('Precision : %.2f' % precision)
    
def grafik_3d (colors, train_X):
    fig = px.scatter_3d(
        train_X, x=0, y=1, z=2,
        color=(colors), 
        labels={'color': 'Bankrupcy?'}
    )
    fig.update_traces(marker_size=1)
    fig.show()
    
def grafik_2d (train_X,train_y):
    plt.scatter(train_X[:,0][train_y==0], train_X[:,1][train_y==0], s=5, c='b')
    plt.scatter(train_X[:,0][train_y==1], train_X[:,1][train_y==1], s=5, c='r')
    plt.show()
    
def plot_ksd(explained_variance_ratio_):    
    plt.figure(figsize = (6,3))
    plt.plot(np.cumsum(explained_variance_ratio_))
    plt.xlabel('Broj glavnih komponenti')
    plt.ylabel('Kumulativna suma disperzije')
    
def plot_uud(explained_variance_ratio_):
    plt.figure(figsize = (6,3))
    plt.plot(explained_variance_ratio_, "bo")
    plt.xlabel('Redni broj glavne komponente')
    plt.ylabel('Disperzija')
    
def najcesca_predikcija(pred_y, broj_predikcija):
    return 1*(sum(pred_y) > (broj_predikcija-1)/2)


X = df.iloc[:,1:96]
y = df.iloc[:,0]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=420, stratify=y)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_X, train_y);
pred_y = model.predict(test_X)
cm = confusion_matrix(test_y, pred_y)
plot_cm(cm)

train_X_jesu_bankrotirali = train_X[train_y == 1]
train_X_nisu_bankrotirali = train_X[train_y == 0]
train_y_jesu_bankrotirali = train_y[train_y == 1] 
train_y_nisu_bankrotirali = train_y[train_y == 0] 
n = len(train_y[train_y == 1])
N = len(train_y[train_y == 0])
N_list = list(np.arange(1, N))
idx = sample(N_list,n) 
train_X_nisu_bankrotirali_us = train_X_nisu_bankrotirali.iloc[idx,]
train_y_nisu_bankrotirali_us = train_y_nisu_bankrotirali.iloc[idx]
train_X_us = pd.concat([train_X_jesu_bankrotirali, train_X_nisu_bankrotirali_us])
train_y_us = pd.concat([train_y_jesu_bankrotirali, train_y_nisu_bankrotirali_us])

model = KNeighborsClassifier(n_neighbors=15)
model.fit(train_X_us, train_y_us);
pred_y = model.predict(test_X)
cm = confusion_matrix(test_y, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

broj_trening_skupova = N//n 
broj_trening_skupova
train_X_us2= {}
train_y_us2= {}
i = 0
while i < 29:
    train_X_tt = train_X_nisu_bankrotirali.iloc[i*n:(i+1)*n]
    train_y_tt = train_y_nisu_bankrotirali.iloc[i*n:(i+1)*n]
    
    train_X_t = pd.concat([train_X_jesu_bankrotirali, train_X_tt])
    train_y_t = pd.concat([train_y_jesu_bankrotirali, train_y_tt])
    
    train_X_us2[i] = train_X_t
    train_y_us2[i] = train_y_t
    
    i += 1

model = KNeighborsClassifier(n_neighbors=15)

i = 0
model.fit(train_X_us2[i], train_y_us2[i]);
pred_y_m = np.array([model.predict(test_X)])

i = 1

while i < 29:
    model.fit(train_X_us2[i], train_y_us2[i])
    pred_y_= np.array([model.predict(test_X)])
    pred_y_m = np.vstack([pred_y_m, pred_y_])
    i += 1

pred_y = najcesca_predikcija(pred_y_m[:,], broj_trening_skupova)
cm = confusion_matrix(test_y, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)


smote = SMOTE(random_state=42)
train_X_smote, train_y_smote = smote.fit_resample(train_X, train_y)

model = KNeighborsClassifier(n_neighbors=150)
model.fit(train_X_smote, train_y_smote)
pred_y = model.predict(test_X)
precision = [metrics.precision_score(test_y, pred_y)]
cm = confusion_matrix(test_y, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

X = df.iloc[:,1:96]
y = df.iloc[:,0]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
pca = PCA().fit(train_X)
plot_ksd(pca.explained_variance_ratio_)
pca = PCA(n_components=0.99).fit(train_X)
n = pca.n_components_ 
n
plot_uud(pca.explained_variance_ratio_)

train_X_pca = pca.fit_transform(train_X)
test_X_pca = pca.transform(test_X)
classifier = LogisticRegression(random_state = 4)
classifier.fit(train_X_pca, train_y)
pred_y = classifier.predict(test_X_pca)
cm = confusion_matrix(test_y, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

train_X_pca2, train_y_pca2 = smote.fit_resample(train_X_pca, train_y)
model = KNeighborsClassifier(n_neighbors=150)
model.fit(train_X_pca2, train_y_pca2);
pred_y = model.predict(test_X_pca)
cm = confusion_matrix(test_y, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

umap_ = umap.UMAP(n_components=2)
X_umap = umap_.fit_transform(X)
train_X_umap, test_X_umap, train_y_umap, test_y_umap = train_test_split(X_umap,y,test_size=0.3, random_state=42 )
grafik_2d(train_X_umap, train_y_umap)

smote = SMOTE(random_state=42)
train_X_umap2, train_y_umap2 = smote.fit_resample(train_X_umap, train_y_umap)
grafik_2d(train_X_umap2,train_y_umap2)
model = KNeighborsClassifier(n_neighbors=50)
model.fit(train_X_umap2, train_y_umap2);
pred_y = model.predict(test_X_umap)
cm = confusion_matrix(test_y_umap, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y_umap, pred_y)

umap_ = umap.UMAP(n_components=3)
X_umap3 = umap_.fit_transform(X)
train_X_umap3, test_X_umap3, train_y_umap3, test_y_umap3 = train_test_split(X_umap3,y,test_size=0.3, random_state=42 )
colors = train_y_umap3.astype(str)
grafik_3d(colors, train_X_umap3)

smote = SMOTE(random_state=42)
train_X_umap4, train_y_umap4 = smote.fit_resample(train_X_umap3, train_y_umap3)
colors = train_y_umap4.astype(str)
grafik_3d(colors, train_X_umap4)
model = KNeighborsClassifier(n_neighbors=50)
model.fit(train_X_umap4, train_y_umap4);
pred_y = model.predict(test_X_umap3)
cm = confusion_matrix(test_y_umap3, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y_umap3, pred_y)

pca = PCA(n_components=0.99).fit(X)
X_pca = pca.fit_transform(X)
umap_ = umap.UMAP(n_components=2)
X_umap5 = umap_.fit_transform(X_pca)
train_X_umap5, test_X_umap5, train_y_umap5, test_y_umap5 = train_test_split(X_umap5,y,test_size=0.3, random_state=42 )
grafik_2d(train_X_umap5, train_y_umap5)
smote = SMOTE(random_state=42)
train_X_umap6, train_y_umap6 = smote.fit_resample(train_X_umap5, train_y_umap5)
grafik_2d(train_X_umap6, train_y_umap6)
model = KNeighborsClassifier(n_neighbors=50)
model.fit(train_X_umap6, train_y_umap6);
pred_y = model.predict(test_X_umap5)
cm = confusion_matrix(test_y_umap5, pred_y)
plot_cm(cm)
mere_kvaliteta(test_y_umap5, pred_y)

model_forest = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, random_state=7)
model_forest.fit(train_X, train_y)
pred_y = model_forest.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)

model_forest = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, random_state=7)
model_forest.fit(train_X_smote, train_y_smote)
pred_y = model_forest.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

svc = svm.SVC(C=100, kernel = 'rbf').fit(train_X, train_y)
pred_y = svc.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)

svc = svm.SVC(C=100, kernel = 'rbf').fit(train_X_smote, train_y_smote)
pred_y = svc.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

svc = svm.SVC(C=100, kernel = 'rbf').fit(train_X_pca2, train_y_pca2)
pred_y = svc.predict(test_X_pca)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

svc = svm.SVC(C=100, kernel = 'poly').fit(train_X_smote, train_y_smote)
pred_y = svc.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

svc = svm.SVC(C=100, kernel = 'poly').fit(train_X_pca2, train_y_pca2)
pred_y = svc.predict(test_X_pca)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=5)
xgb.fit(train_X, train_y)
pred_y = xgb.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=5)
xgb.fit(train_X_smote, train_y_smote)
pred_y = xgb.predict(test_X)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)

xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=5)
xgb.fit(train_X_pca2, train_y_pca2)
pred_y = xgb.predict(test_X_pca)
cm = confusion_matrix(test_y,pred_y)
plot_cm(cm)
mere_kvaliteta(test_y, pred_y)