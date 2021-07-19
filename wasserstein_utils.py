import numpy as np

import scipy.io as sio

import tensorflow as tf
import tensorflow.keras.utils
import tensorflow.keras.backend as K

def generateTheta(L,endim):
    theta_=np.random.normal(size=(L,endim))
    for l in range(L):
        theta_[l,:]=theta_[l,:]/np.sqrt(np.sum(theta_[l,:]**2))
    return theta_

def oneDWassersteinV3(p,q):
    # ~10 Times faster than V1

    psort=tf.sort(p,axis=0)
    qsort=tf.sort(q,axis=0)
    pqmin=tf.minimum(K.min(psort,axis=0),K.min(qsort,axis=0))
    psort=psort-pqmin
    qsort=qsort-pqmin
    
    n_p=tf.shape(p)[0]
    n_q=tf.shape(q)[0]
    
    pcum=tf.multiply(tf.cast(tf.maximum(n_p,n_q),dtype='float32'),tf.divide(tf.cumsum(psort),tf.cast(n_p,dtype='float32')))
    qcum=tf.multiply(tf.cast(tf.maximum(n_p,n_q),dtype='float32'),tf.divide(tf.cumsum(qsort),tf.cast(n_q,dtype='float32')))
    
    indp=tf.cast(tf.floor(tf.linspace(0.,tf.cast(n_p,dtype='float32')-1.,tf.minimum(n_p,n_q)+1)),dtype='int32')
    indq=tf.cast(tf.floor(tf.linspace(0.,tf.cast(n_q,dtype='float32')-1.,tf.minimum(n_p,n_q)+1)),dtype='int32')
    
    phat=tf.gather(pcum,indp[1:],axis=0)
    phat=K.concatenate((K.expand_dims(phat[0,:],0),phat[1:,:]-phat[:-1,:]),0)
    
    qhat=tf.gather(qcum,indq[1:],axis=0)
    qhat=K.concatenate((K.expand_dims(qhat[0,:],0),qhat[1:,:]-qhat[:-1,:]),0)
          
    W2=K.mean((phat-qhat)**2,axis=0)
    return W2

# Uses less GPU memory than _v1
def sWasserstein_hd_v2(P, Q, theta, nclass, Cp=None, Cq=None):
    # High dimensional variant of the sWasserstein function

    '''
        P, Q - representations in embedding space between target & source
        theta - random matrix of directions
    '''

    tot = 0
    
    for it in range(theta.shape[1]):
        theta1d = K.expand_dims(theta[it])
    
        p = K.dot(K.reshape(P, (-1, nclass)), theta1d)
        q = K.dot(K.reshape(Q, (-1, nclass)), theta1d)
    
        tot += K.mean(oneDWassersteinV3(p,q))
    
    return tot / theta.shape[1]



def sWasserstein_hd(P,Q,theta,nclass,Cp=None,Cq=None):
    # High dimensional variant of the sWasserstein function

    '''
        P, Q - representations in embedding space between target & source
        theta - random matrix of directions
    '''

    p=K.dot(K.reshape(P, (-1, nclass)), K.transpose(theta))
    q=K.dot(K.reshape(Q, (-1, nclass)), K.transpose(theta))
    sw=K.mean(oneDWassersteinV3(p,q))

    return sw

