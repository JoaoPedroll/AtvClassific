from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pickle


dados = pd.read_csv('Cancer_Data.csv',   decimal =',', thousands = '.')

dados.drop(columns=['id','Unnamed: 32','compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],inplace= True)

clf_svm = svm.SVC(gamma=0.001, C=100.0)

y_svm = dados['diagnosis']
X_svm = dados.drop(columns='diagnosis')



arvore_clf = tree.DecisionTreeClassifier()
arvore_clf.fit(X_svm,y_svm)


with open('modelo_treinado.pkl', 'wb') as arquivo:
    pickle.dump(arvore_clf, arquivo)
