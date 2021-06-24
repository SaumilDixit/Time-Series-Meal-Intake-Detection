'''
Todo: 
a)	Extract features from Meal and No Meal data
b)	Make sure that the features are discriminatory
c)	Trains a machine to recognize Meal or No Meal data
d)	Use k fold cross validation on the training data to evaluate your recognition system
e)	Write a function that takes one test sample as input and outputs 1 if it predicts the test sample as meal or 0 if it predicts test sample as No meal.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pywt
#from sympy import fft
import scipy.fftpack
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


cgm = pd.read_csv('CGMData670GPatient1.csv')
ins = pd.read_csv('InsulinAndMealIntake670GPatient1.csv')

col_ins = ['Date','Time','BWZ Carb Input (grams)']
ins_fin = ins[(ins['BWZ Carb Input (grams)'].isnull()==False) & (ins['BWZ Carb Input (grams)'] > 0)][col_ins]
ins_fin.reset_index(drop=True,inplace =True)
ins_fin['datetime'] = pd.to_datetime(ins_fin['Date'] + ' ' + ins_fin['Time'])


cgm_cols = ['Date','Time','Sensor Glucose (mg/dL)']

cgm.dropna(subset = ['Sensor Glucose (mg/dL)'],inplace = True)

cgm['datetime'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'])

cgm_fin = cgm[['Date','Time','Sensor Glucose (mg/dL)','datetime']]

cgm_fin.reset_index(drop=True,inplace =True)
'''
new_data=pd.DataFrame(columns=['datetime','lowtime','uptime','no meal uptime','meal data Sensor Glucose (mg/dL) list'
                               ,'no meal data Sensor Glucose (mg/dL) list'])
'''
new_data=pd.DataFrame(columns=['meal data Sensor Glucose (mg/dL) list','no meal data Sensor Glucose (mg/dL) list'])


for i in range(len(ins_fin)):
    mytime = ins_fin['datetime'].iloc[i]
    uptime = mytime + pd.Timedelta(minutes=120)
    nomeal_uptime = mytime + pd.Timedelta(minutes=240)
    lowtime = mytime - pd.Timedelta(minutes=30)
    myval = list(cgm_fin[(cgm_fin['datetime']>= pd.to_datetime(lowtime)) & (cgm_fin['datetime']<= pd.to_datetime(uptime))]['Sensor Glucose (mg/dL)'])
    mynomeal_val = list(cgm_fin[(cgm_fin['datetime']>= pd.to_datetime(uptime)) & (cgm_fin['datetime']<= pd.to_datetime(nomeal_uptime))]['Sensor Glucose (mg/dL)'])
    #print(myval)
    
    #new_data = new_data.append({'datetime':mytime ,'uptime':uptime,'lowtime':lowtime, 'no meal uptime':nomeal_uptime, 'meal data Sensor Glucose (mg/dL) list':myval, 'no meal data Sensor Glucose (mg/dL) list': mynomeal_val}, ignore_index=True)
    

    new_data = new_data.append({'meal data Sensor Glucose (mg/dL) list':myval,
                                'no meal data Sensor Glucose (mg/dL) list': mynomeal_val}, ignore_index=True)  

final_meal = pd.DataFrame()
    
for i in new_data['meal data Sensor Glucose (mg/dL) list']:
       l=len(i)
       df_x = pd.DataFrame()
       for j in range(l):
           df_x = df_x.append({'meal data':i[j]}, ignore_index=True)
       df_x = df_x.transpose()
       final_meal=final_meal.append(df_x, ignore_index=True)
       del df_x
           
         
final_meal.to_csv('mealDataPat1.csv',index=False, header=None)



final_nomeal = pd.DataFrame()

for i in new_data['no meal data Sensor Glucose (mg/dL) list']:
       l=len(i)
       df_x = pd.DataFrame()
       for j in range(l):
           df_x = df_x.append({'no meal data':i[j]}, ignore_index=True)
       df_x = df_x.transpose()
       final_nomeal=final_nomeal.append(df_x, ignore_index=True)
       del df_x
           
         
final_nomeal.to_csv('noMealDataPat1.csv',index=False, header=None)

cgm3 = pd.read_csv('CGMData670GPatient2.csv')
ins3 = pd.read_csv('InsulinAndMealIntake670GPatient2.csv')

col_ins3 = ['Date','Time','BWZ Carb Input (grams)']
ins3_fin = ins3[(ins3['BWZ Carb Input (grams)'].isnull()==False) & (ins3['BWZ Carb Input (grams)'] > 0)][col_ins3]
ins3_fin.reset_index(drop=True,inplace =True)
ins3_fin['datetime'] = pd.to_datetime(ins3_fin['Date'] + ' ' + ins3_fin['Time'])


cgm3_cols = ['Date','Time','Sensor Glucose (mg/dL)']

cgm3.dropna(subset = ['Sensor Glucose (mg/dL)'],inplace = True)

cgm3['datetime'] = pd.to_datetime(cgm3['Date'] + ' ' + cgm3['Time'])

cgm3_fin = cgm3[['Date','Time','Sensor Glucose (mg/dL)','datetime']]

cgm3_fin.reset_index(drop=True,inplace =True)

#new_data3=pd.DataFrame(columns=['datetime','lowtime','uptime','no meal uptime','meal data Sensor Glucose (mg/dL) list', 'no meal data Sensor Glucose (mg/dL) list'])

new_data3=pd.DataFrame(columns=['meal data Sensor Glucose (mg/dL) list','no meal data Sensor Glucose (mg/dL) list'])


for i in range(len(ins3_fin)):
    mytime = ins3_fin['datetime'].iloc[i]
    uptime = mytime + pd.Timedelta(minutes=120)
    nomeal_uptime = mytime + pd.Timedelta(minutes=240)
    lowtime = mytime - pd.Timedelta(minutes=30)
    myval = list(cgm3_fin[(cgm3_fin['datetime']>= pd.to_datetime(lowtime)) & (cgm3_fin['datetime']<= pd.to_datetime(uptime))]['Sensor Glucose (mg/dL)'])
    mynomeal_val = list(cgm3_fin[(cgm3_fin['datetime']>= pd.to_datetime(uptime)) & (cgm3_fin['datetime']<= pd.to_datetime(nomeal_uptime))]['Sensor Glucose (mg/dL)'])
    #print(myval)
    
    #new_data3 = new_data3.append({'datetime':mytime ,'uptime':uptime,'lowtime':lowtime, 'no meal uptime':nomeal_uptime, 'meal data Sensor Glucose (mg/dL) list':myval, 'no meal data Sensor Glucose (mg/dL) list': mynomeal_val}, ignore_index=True)
    

    new_data3 = new_data3.append({'meal data Sensor Glucose (mg/dL) list':myval,
                                'no meal data Sensor Glucose (mg/dL) list': mynomeal_val}, ignore_index=True)  

final_meal3 = pd.DataFrame()
    
for i in new_data3['meal data Sensor Glucose (mg/dL) list']:
       l=len(i)
       df_x = pd.DataFrame()
       for j in range(l):
           df_x = df_x.append({'meal data':i[j]}, ignore_index=True)
       df_x = df_x.transpose()
       final_meal3=final_meal3.append(df_x, ignore_index=True)
       del df_x
           
         
final_meal3.to_csv('mealDataPat2.csv',index=False, header=None)



final_nomeal3 = pd.DataFrame()

for i in new_data3['no meal data Sensor Glucose (mg/dL) list']:
       l=len(i)
       df_x = pd.DataFrame()
       for j in range(l):
           df_x = df_x.append({'no meal data':i[j]}, ignore_index=True)
       df_x = df_x.transpose()
       final_nomeal3=final_nomeal3.append(df_x, ignore_index=True)
       del df_x
           
         
final_nomeal3.to_csv('noMealDataPat2.csv',index=False, header=None)

def main():
    cgmseries, classes = readData()
    normalisedFeatureMat = calculateFeatures(cgmseries)
    
    pca1 = PCA().fit(normalisedFeatureMat.data)
    plt.plot(np.cumsum(pca1.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    pca = PCA(n_components=10)
    components = pca.fit_transform(normalisedFeatureMat)


    #print(pca.components_)
    #print(pca.explained_variance_)
    #print(pca.explained_variance_ratio_)


    X_train, X_test, y_train, y_test = train_test_split(components, classes, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(n_estimators=260)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print("Confusion Matrix:", metrics.confusion_matrix(y_test,y_pred))
    #print("Classification Report:", metrics.classification_report(y_test,y_pred))
    pickle.dump(classifier, open('classifier_model.pkl', 'wb'))
    pickle.dump(pca, open('pca_model.pkl', 'wb'))



def readData():
    cgmseries = pd.read_csv('mealDataPat1.csv', usecols=list(range(24)) , names=list(range(24)) )
    classes = []
    nextfile = pd.read_csv('mealDataPat2.csv', usecols=list(range(24)) , names=list(range(24)))
    cgmseries = cgmseries.append(nextfile)


    cgmseries = cgmseries.dropna()
    classes.extend([1]*cgmseries.shape[0])


    for i in range(1,3):
        nextfile = pd.read_csv('noMealDataPat'+str(i)+'.csv', usecols=list(range(24)) , names=list(range(24)))
    #nextfile = pd.read_csv('noMealDataPat1.csv', usecols=list(range(24)) , names=list(range(24)))
    alteredfile = nextfile.dropna()
    cgmseries = cgmseries.append(alteredfile)
    classes.extend([0]*alteredfile.shape[0])

    #cgmseries.insert(loc=30, value=classes)
    #print(cgmseries.head())
    numrows = cgmseries.shape[0]
    numcols = cgmseries.shape[1]
    #print(numrows)
    #print(numcols)
    #print(len(classes))
    #cgmseries = cgmseries.astype('float')
    return cgmseries, classes



def calculateFeatures(cgmseries):
    numrows = cgmseries.shape[0]
    numcols = cgmseries.shape[1]
    #Kurtosis

    kurtosis = []
    for i in range(numrows):
        kurtosis.append(cgmseries.iloc[i,:].kurtosis())

    #Dicrete Wavelet transform
    dwtCoeff = []
    for i in range(numrows):
        #Perform DWT using Debauchies wavelet for two levels of transformation
        coeffs = pywt.wavedec(cgmseries.iloc[i,:], 'db2', level=2) 
        cA2, cD2, cD1 = coeffs
        #print(len(cD2))
        dwtCoeff.append(cD2)
    #print(len(dwtCoeff))

    #Time in Range
    tircounter = np.zeros(numrows)
    for i in range(numrows):
        for j in range(numcols):
            if(cgmseries.iloc[i,j]>=70 and cgmseries.iloc[i,j]<=180):
                tircounter[i] = tircounter[i] + 1
        tircounter[i] = tircounter[i]/(numcols)

    #LAGE
    LAGE = np.zeros(numrows)
    for i in range(numrows):
        LAGE[i] = np.max(cgmseries.iloc[i,:]) - np.min(cgmseries.iloc[i,:])
    #print(len(LAGE))

    #FFT
    fftCoeff = []
    for i in range(numrows):
        nfft = np.fft.fft(cgmseries.iloc[i,:])
        nfft = np.absolute(nfft)
        #print(nfft[0:2])
        fftCoeff.append(nfft[0:8])
    #print(len(fftCoeff))

    #Poly COeff
    d = 7
    polyCoeff = np.zeros((numrows,d+1))
    for i in range(numrows): 
        polyCoeff[i,:] = np.polyfit(range(0,120,5),cgmseries.iloc[i,:], deg=d)
    #print(polyCoeff)

    featureMat = [0]*numrows
    for i in range(numrows):
        featureMat[i] = np.concatenate((polyCoeff[i], fftCoeff[i], dwtCoeff[i], np.asarray([tircounter[i], kurtosis[i], LAGE[i]])))

    #print(len(featureMat))
    #print(len(featureMat[0]))
    featureMat = np.asarray(featureMat)
    normalisedMat = StandardScaler().fit_transform(featureMat)
    #print(normalisedMat)
    #print(np.mean(normalisedMat),np.std(normalisedMat))
    return normalisedMat

if __name__ == '__main__':
    main()
