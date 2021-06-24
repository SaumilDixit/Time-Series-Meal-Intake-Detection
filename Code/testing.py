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
import sys

def readData(filePath):

    cgmseries = pd.read_csv(filePath, usecols=list(range(24)) , names=list(range(24)) )
    cgmseries = cgmseries.dropna()
    #print(cgmseries.head())
    #print(cgmseries.shape)
    return cgmseries


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


def main():
    
    #Usage: python testing.py *filepath*
    cgmseries = readData(sys.argv[1])

    normalisedFeatureMat = calculateFeatures(cgmseries)

    pca = pickle.load(open('pca_model.pkl', 'rb'))
    components = pca.transform(normalisedFeatureMat)

    classifier = pickle.load(open('classifier_model.pkl', 'rb'))
    y_pred = classifier.predict(components)
    #print(y_pred)

    #Storing vector in Results.csv file
    res = pd.DataFrame(y_pred)
    #res.columns = ['Class Label']
    res.to_csv('Results.csv', index=False, header=None)

if __name__ == '__main__' :
    main()
