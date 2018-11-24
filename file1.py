import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import tree
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn import metrics
import itertools

df = pd.read_csv("/Users/anushka/Desktop/IDS/Project/diabetes.csv")
print(df.shape)
df.head()

df.describe()

#find no. of rows with missing values of relevent attribute
(df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]==0).sum()

#fill the missing values with NaNs to make them identifiable
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.head()

#missing values are set to the average of all values of that attribute
values = df.values
df.fillna(df.mean(), inplace=True)
df.head()

#visualizing the dataset

outcome = [df[df['Outcome']==1].count()['Glucose'],df[df['Outcome']==0].count()['Glucose']]
label = ['Diabetic', 'Non-diabetic']
colors = ['r', 'b']
plt.pie(outcome, labels=label, colors=colors, startangle=90, autopct='%.2f%%')
plt.show()

#pregnancies
plt.figure
x1 = np.arange(max(df['Pregnancies'])+1)
y1 = []
for i in range(len(x1)):
    y1.append(df[df['Pregnancies']==x1[i]].count()[0])
plt.bar(x1,y1)
plt.xticks(x1)
plt.xlabel('No. of pregnancies')
plt.show()

#glucose
plt.figure
x2 = df['Glucose'].values
plt.hist(x2,bins=20)
plt.xlabel('Glucose levels')
plt.show()

#blood pressure
plt.figure
x3 = df['BloodPressure'].values
plt.hist(x3,bins=20)
plt.xlabel('Blood Pressure Levels')
plt.show()

#Skin Thickness
plt.figure
x4 = df['SkinThickness'].values
plt.hist(x4,bins=10)
plt.xlabel('Skin Thickness')
plt.show()

#insulin
plt.figure
x5 = df['Insulin'].values
plt.hist(x5,bins=10)
plt.xlabel('Insulin levels')
plt.show()

#bmi
plt.figure
x6 = df['BMI'].values
plt.hist(x6,bins=20)
plt.xlabel('BMI')
plt.show()

#dpf
plt.figure
x7 = df['DiabetesPedigreeFunction'].values
plt.hist(x7,bins=20)
plt.xlabel('dpf values')
plt.show()

#age
plt.figure
x8 = df['Age'].values
plt.hist(x8,bins=20)
plt.xlabel('Age')
plt.show() 

X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['Outcome']
X_norm = (X - X.min())/(X.max()-X.min())
X_train,X_test,y_train,y_test = train_test_split(X_norm, y, test_size=0.33, random_state=0)
X_train.shape

outcome = [y_train[y_train==1].count(),y_train[y_train==0].count()]
label = ['Diabetic', 'Non-diabetic']
colors = ['r', 'b']
plt.pie(outcome, labels=label, colors=colors, startangle=90, autopct='%.1f%%')
plt.show()
y_train[y_train==0].count()

plt.figure
sns.set(style='ticks')
sns.pairplot(df, hue='Outcome')
plt.show()

pca = sklearnPCA(n_components=2) #2-dimensional PCA
pca_transformed = pd.DataFrame(pca.fit_transform(X_norm))

xax = pca_transformed[0]
yax = pca_transformed[1]
plt.scatter(xax[y==1], yax[y==1], label='Diabetic', c='red')
plt.scatter(xax[y==0], yax[y==0], label='Non-diabetic', c='blue')
plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

clf1 = SVC(kernel='rbf', gamma=0.1, C=1)  
clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)
sc1 = clf1.score(X_test,y_test)
f1 = f1_score(y_test,pred1)
print('Accuracy: '+str(sc1))
print('F-score '+str(f1))
cm = confusion_matrix(y_test,pred1,labels=[0,1])
plot_confusion_matrix(cm,classes=[0,1])

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)
sc2 = clf2.score(X_test,y_test)
f2 = f1_score(y_test,pred2)
print('Accuracy: '+str(sc2))
print('F-score '+str(f2))
cm = confusion_matrix(y_test,pred2,labels=[0,1])
plot_confusion_matrix(cm,classes=[0,1])

clf3 = RandomForestClassifier(max_depth=6, n_estimators=8, random_state=0)
clf3.fit(X_train,y_train)
pred3 = clf3.predict(X_test)
sc3 = clf3.score(X_test,y_test)
f3 = f1_score(y_test,pred3)
print('Accuracy: '+str(sc3))
print('F-score '+str(f3))
cm = confusion_matrix(y_test,pred3,labels=[0,1])
plot_confusion_matrix(cm,classes=[0,1])