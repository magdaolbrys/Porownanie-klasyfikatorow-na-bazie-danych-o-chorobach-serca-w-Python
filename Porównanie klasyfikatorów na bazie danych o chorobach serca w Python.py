import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
d =  pd.read_csv('heart_disease_uci.csv')
d = d.dropna()


#zamiana zmiennych na kategoryczne
d['sex'] = d['sex'].replace('Male',0)
d['sex'] = d['sex'].replace('Female',1)

d['dataset'] = d['dataset'].replace('Cleveland',1)
d['dataset'] = d['dataset'].replace('Hungary',2)
d['dataset'] = d['dataset'].replace('VA Long Beach',3)
d['dataset'] = d['dataset'].replace('Switzerland',4)

d['cp'] = d['cp'].replace('typical angina',1)
d['cp'] = d['cp'].replace('atypical angina',2)
d['cp'] = d['cp'].replace('non-anginal',3)
d['cp'] = d['cp'].replace('asymptomatic',4)

d['fbs'] = (d['fbs']).astype(int)

d['restecg'] = d['restecg'].replace('normal',1)
d['restecg'] = d['restecg'].replace('st-t abnormality',2)
d['restecg'] = d['restecg'].replace('lv hypertrophy',3)

d['exang'] = (d['exang']).astype(int)

d['slope'] = d['slope'].replace('upsloping',1)
d['slope'] = d['slope'].replace('flat',2)
d['slope'] = d['slope'].replace('downsloping',3)

d['thal'] = d['thal'].replace('normal',1)
d['thal'] = d['thal'].replace('fixed defect',2)
d['thal'] = d['thal'].replace('reversable defect',3)

d['num'] = d['num'].replace(2,1)
d['num'] = d['num'].replace(3,1)
d['num'] = d['num'].replace(4,1)



print('minimalny wiek:' ,min(d['age']))
print('maksymalny wiek: ',max(d['age']))
print('średni wiek: ',round(np.mean(d['age'])))
print('minimalne ciśnienie krwi:' ,min(d['trestbps']))
print('maksymalne ciśnienie krwi: ',max(d['trestbps']))
print('średnie ciśnienie krwi: ',round(np.mean(d['trestbps'])))
print('minimalny poziom cholesterolu:' ,min(d['chol']))
print('maksymalny poziom cholesterolu: ',max(d['chol']))
print('średni poziom cholesterolu: ',round(np.mean(d['chol'])))
print('minimalne tętno:' ,min(d['thalch']))
print('maksymalne tętno: ',max(d['thalch']))
print('średnie tętno: ',round(np.mean(d['thalch'])))
print('minimalne obniżenie ST wywołane wysiłkiem fizycznym w stosunku do spoczynku:' ,min(d['oldpeak']))
print('maksymalne obniżenie ST wywołane wysiłkiem fizycznym w stosunku do spoczynku: ',max(d['oldpeak']))
print('średnie obniżenie ST wywołane wysiłkiem fizycznym w stosunku do spoczynku: ',round(np.mean(d['oldpeak'])))


#age
under50 = 0
uppereq50 = 0
age = (d['age'])
for i in age:
    if i < 50:
        under50 = under50+1
    else:
        uppereq50 = uppereq50 +1
v = ('mniej niż 50 lat', '50 lub więcej lat')
l = (under50,uppereq50)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Wiek badanych:', size = 15)
plt.show()

#sex
male = sum(d['sex'])
female = len(d) - male
v = ('kobieta', 'mężczyzna')
l = (male,female)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Płeć badanych:', size = 15)
plt.show()


#dataset
C = 0
H = 0
V = 0
S =0
dataset = (d['dataset'])
for i in dataset:
    if i == 1:
        C = C + 1
    elif i == 2:
        H = H + 1
    elif i == 3:
        V = V + 1
    elif i == 4:
        S = S + 1
v = ('Cleveland', 'Hungary', 'VALongBeach', 'Switzerland')
l = (C,H,V,S)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Pochodzenie badanych:', size = 15)
plt.show()



#cp
ta = 0
aa = 0
na = 0
a =0
cp = (d['cp'])
for i in cp:
    if i == 1:
        ta = ta + 1
    elif i == 2:
        aa = aa + 1
    elif i == 3:
        na = na + 1
    elif i == 4:
        a = a + 1
v = ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic')
l = (ta,aa,na,a)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Typ bólu w klatce piersiowej:', size = 15)
plt.show()



#trestbps
morethan120 = 0
eqless = 0
trestbps = (d['trestbps'])
for i in trestbps:
    if i <= 120:
        eqless = eqless + 1
    elif i > 120:
        morethan120  = morethan120 +1

v = ('powyzej normy', 'w normie')
l = (morethan120,eqless)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Cisnienie krwi badanych:', size = 15)
plt.show()


#chol 
good = 0
bad = 0
for i in d['chol']:
    if i >= 190:
        bad = bad + 1
    elif i < 190:
        good  = good +1
v = ('w normie', 'powyżej normy')
l = (good,bad)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Poziom cholesterolu badanych:', size = 15)
plt.show()

#fbs 
a = 0
b = 0
for i in d['fbs']:
    if i == 0:
        a = a + 1
    elif i == 1:
        b = b + 1
v = ('w normie', 'powyżej normy')
l = (a,b)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Poziom cukru we krwi na czczo:', size = 15)
plt.show()



#restecg 
a = 0
b = 0
c = 0
for i in d['restecg']:
    if i == 1:
        a = a + 1
    elif i == 2:
        b = b + 1
    elif i == 3:
        c = c + 1
v = ('normal', 'st-t abnormality', 'lv hypertrophy')
l = (a,b,c)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Wynik spoczynkowego EKG:', size = 15)
plt.show()


#thalach: 
a = 0
b = 0
c = 0

for i in d['thalch']:
    if i <60:
        a = a+1
    elif i > 100:
        b = b+1
    else:
        c = c+1

v = ('ponizej normy', 'w normie','powyzej normy')
l = (a,b,c)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Tętno badanych:', size = 15)
plt.show()

#exang: 
a = 0
b = 0
for i in d['exang']:
    if i == 0:
        a = a + 1
    elif i == 1:
        b = b + 1
v = ('nie występuje', 'występuje')
l = (a,b)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Angina wywoływana wysiłkiem fizycznym :', size = 15)
plt.show()
    
#oldpeak: 
a = 0
b = 0
for i in d['oldpeak']:
    if i >= 1:
        a = a+1
    else:
        b = b+1
v = ('wieksze lub równe 1', 'mniejsze niż 1')
l = (a,b)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Obniżenie ST wywołane wysiłkiem fizycznym w stosunku do odpoczynku :', size = 15)
plt.show()  

#slope: 
a = 0
b = 0
c = 0
for i in d['slope']:
    if i == 1:
        a = a + 1
    elif i == 2:
        b = b + 1
    elif i == 3:
        c = c + 1
v = ('upsloping', 'flat','downsloping')
l = (a,b,c)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Nachylenie szczytowego odcinka ST podczas wysiłku:', size = 15)
plt.show()

#ca:    
a = 0
b = 0
c = 0
e = 0
for i in d['ca']:
    if i == 0:
        a = a + 1
    elif i == 1:
        b = b + 1
    elif i == 2:
        c = c + 1
    elif i == 3:
        e = e + 1
v = ('0', '1','2','3')
l = (a,b,c,e)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Liczba głównych naczyń zabarwionych przy fluoroskopii:', size = 15)
plt.show()


#thal:
a = 0
b = 0
c = 0
for i in d['thal']:
    if i == 1:
        a = a + 1
    elif i == 2:
        b = b + 1
    elif i == 3:
        c = c + 1
v = ('normal', 'fixed defect','reversable defect')
l = (a,b,c)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Talasemia:', size = 15)
plt.show()    

#num:  
a = 0
b = 0
for i in d['num']:
    if i == 0:
        a = a + 1
    elif i == 1:
        b = b + 1     
v = ('tak', 'nie')
l = (b,a)
plt.bar(v, l)
plt.xticks(v,rotation = 0)
plt.ylabel('liczba badanych')
plt.title('Choroba serca:', size = 15)
plt.show()  
 
    

from sklearn.model_selection import train_test_split
import sklearn as sk
#dziele zbiory na uczący i testowy
x = d[['age','sex','dataset','cp','trestbps','chol','fbs','restecg','thalch','exang',
       'oldpeak','slope','ca','thal']]
y = d[['num']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
#tree
t = sk.tree.DecisionTreeClassifier() #budowa modelu
t.fit(x_train, y_train) #trenowanie modelu
score_t = t.score(x_test, y_test) 
m = plot_confusion_matrix(t, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - Decision Tree') 
plt.show() #wyswietlam

#gaussian naive bayes
from sklearn.naive_bayes import GaussianNB
nv = GaussianNB() # klasyfikator
nv.fit(x_train, y_train.values.ravel()) #trenowanie modelu
score_nv = nv.score(x_test, y_test) 
m = plot_confusion_matrix(nv, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - GaussianNB') 
plt.show() #wyswietlam

from sklearn.neighbors import KNeighborsClassifier
#KNN k = 1
k1= KNeighborsClassifier(n_neighbors=1) # klasyfikator
k1.fit(x_train, y_train.values.ravel()) #trenowanie modelu
score_k1 = k1.score(x_test, y_test) 
m = plot_confusion_matrix(k1, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - 1nearest neighbor') 
plt.show() #wyswietlam

#KNN k = 3
k3= KNeighborsClassifier(n_neighbors=3) # klasyfikator
k3.fit(x_train, y_train.values.ravel()) #trenowanie modelu
score_k3 = k3.score(x_test, y_test) 
m = plot_confusion_matrix(k3, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - 3 nearest neighbors') 
plt.show() #wyswietlam

#KNN k = 5
k5= KNeighborsClassifier(n_neighbors=5) # klasyfikator
k5.fit(x_train, y_train.values.ravel()) #trenowanie modelu
score_k5 = k5.score(x_test, y_test) 
m = plot_confusion_matrix(k5, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - 5 nearest neighbors') #dodaje tytuł
plt.show() #wyswietlam

#KNN k = 8
k8= KNeighborsClassifier(n_neighbors=8) # klasyfikator
k8.fit(x_train, y_train.values.ravel()) #trenowanie modelu
score_k8 = k8.score(x_test, y_test) 
m = plot_confusion_matrix(k8, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - 8 nearest neighbors') #dodaje tytuł
plt.show() #wyswietlam

#randomforest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier() #budowa modelu
rf.fit(x_train, y_train) #trenowanie modelu
score_rf = rf.score(x_test, y_test) 
m = plot_confusion_matrix(rf, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - Random Forest') #dodaje tytuł
plt.show() #wyswietlam

#regresja logistyczna  
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
score_lr = lr.score(x_test, y_test) 
m = plot_confusion_matrix(lr, x_test, y_test) #tworze macierz
m.ax_.set_title('Confusion matrix - LogisticRegression') #dodaje tytuł
plt.show() #wyswietlam

s = [] #lista z wynikami score dla każdego modelu
s.append(score_t)
s.append(score_nv)
s.append(score_k1)
s.append(score_k3)
s.append(score_k5)
s.append(score_k8)
s.append(score_rf)
s.append(score_lr)
#wykres
models = ('Decision Tree', 'Gaussian Naive Bayes', 'Nearest Neighbours k=1', 
          'Nearest Neighbours k=3', 'Nearest Neighbours k=5','Nearest Neighbours k=8',
          'Random Forest','Logistic Regression')
l = np.arange(len(models))
plt.bar(l, s, color = ('bisque', 'orange', 'darkorange','darkorange','darkorange',
                       'darkorange','coral','peachpuff'),alpha = 0.7)
plt.xticks(l, models,rotation = 90)
plt.ylabel('Score')
plt.title('Score(accuracy) of the classifiers:', size = 15)
sc = (round(score_t,2), round(score_nv,2),round(score_k1,2), round(score_k3,2),
      round(score_k5,2),round(score_k8,2),round(score_rf,2),round(score_lr,2))
for i in range(len(sc)):
    plt.text(x = i-0.2 , y = sc[i], s = sc[i], size = 10)
plt.show()


