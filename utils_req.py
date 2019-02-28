import numpy as np
import decimal
import matplotlib.pyplot as plt
from scipy import interpolate as intp
from scipy import stats
import seaborn as sns
from itertools import compress
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import *
from scipy import signal
from scipy import stats
from scipy.signal import convolve as sig_convolve
from scipy.io import loadmat
from scipy.io import savemat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from pathlib import Path
import cv2
import time
import os.path




def pick_indices(x,l,u):
    """
    Pick the elements of an array x that lie between l and u
    """
    id_s = np.nonzero(x >= l)[0]
    id_e = np.nonzero(x <= u)[0]
    return np.intersect1d(id_s,id_e)

def vonMisesFn(x,amp,scale,theta0):
    """
    Von Mises function, but for orientations
    x goes from (0, pi)
    """
    return amp*np.exp(scale*np.cos(2*(x-theta0)))

def expandN_dims(x,N):
    """
    Add N extra dimensions to x
    """
    dims = np.shape(x.shape)[0] 
    for i in range(N):
        x = np.expand_dims(x,axis=dims+i)
    return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def findRFcenter(frame,ksize,sigma_o):
    """
    Function to find the center of the receptive field 
    """
    gx = frame[:,1:] - frame[:,0:-1]
    gx = np.concatenate((gx,np.expand_dims(gx[:,-1],axis=1)),axis=1)

    gy = frame[1:,:] - frame[0:-1,:]
    gy = np.concatenate((gy,np.expand_dims(gy[-1,:],axis=0)),axis=0)

    gxx = gx**2
    gyy = gy**2

    gxx = cv2.GaussianBlur(gxx,(ksize,ksize),sigma_o)
    gyy = cv2.GaussianBlur(gyy,(ksize,ksize),sigma_o)

    rmap = np.sqrt(gxx + gyy)

    max_x = np.max(np.abs(rmap),axis=1)
    id_x = np.argmax(np.abs(rmap),axis=1)
    Ry = np.argmax(max_x)
    Rx = id_x[Ry]
    
    return Rx, Ry


def ProcessReceptiveFields(ReceptiveFields):
    """
    Function to process the receptive fields
    Expects raw STAs as input. Input dimension: NNeu x Ny x Nx
    1. Normalize the RFs for each neuron such that the absolute value peak is at 1
    2. Find the center for each RF
    3. Compute the SNR metric to assess the quality of the receptive field
    """
    
    NNeu, Ny, Nx = ReceptiveFields.shape
    
    # 1. Normalize by the abs max
    RFPeaks = np.max(np.abs(ReceptiveFields.reshape(NNeu,-1)),axis=1)
    ReceptiveFields = ReceptiveFields/expandN_dims(RFPeaks,2)
    
    # 2. Find the RF centers
    RxVec = np.zeros([NNeu])
    RyVec = np.zeros([NNeu])
    ksize = 9    # Gaussian kernel size  used for smoothing RF before finding center
    sigma_o = 3  # and its standard deviation 
    for k in range(NNeu):
        RxVec[k], RyVec[k] = findRFcenter(ReceptiveFields[k],ksize,sigma_o)
    
    # 3. Compute the snr metric
    snr = np.max(np.abs(ReceptiveFields.reshape(NNeu,-1)),axis=1)/np.std(ReceptiveFields.reshape(NNeu,-1),axis=1)
    
    return ReceptiveFields, np.uint8(RxVec), np.uint8(RyVec), snr


def Extract_Orientation_Contrast(frame, ksize, sigma_o, sigma_c, rs):
    """
    Function that estimates the local orientation and contrast at each pixel in the input image frame.
    Inputs:
    frame   - input image
    ksize   - kernel size of the gaussian kernel used to smooth the squared gradients
    sigma_o - standard deviation of the gaussian kernel for estimating orientations
    sigma_c - standard deviation of the gaussian kernel for estimating orientations
    rs      - factor by which to resize the gradient functions
    
    Outputs: 
    Ohat  - estimated orientation
    Chat  - estimated contrast
    """

    # First, compute the gradients
    gx = frame[:,1:] - frame[:,0:-1]
    gx = np.concatenate((gx,np.expand_dims(gx[:,-1],axis=1)),axis=1)

    gy = frame[1:,:] - frame[0:-1,:]
    gy = np.concatenate((gy,np.expand_dims(gy[-1,:],axis=0)),axis=0)

    gxx = gx**2
    gyy = gy**2
    gxy = gx*gy

    # smoothen these sqaured gradients if needed
    gxx = cv2.GaussianBlur(gxx,(ksize,ksize),sigma_o)
    gyy = cv2.GaussianBlur(gyy,(ksize,ksize),sigma_o)
    gxy = cv2.GaussianBlur(gxy,(ksize,ksize),sigma_o)

    # resize these gradient functions
    if rs != 1:
        gxx = cv2.resize(gxx,None,fx=rs, fy=rs, interpolation = cv2.INTER_LANCZOS4)
        gyy = cv2.resize(gyy,None,fx=rs, fy=rs, interpolation = cv2.INTER_LANCZOS4)
        gxy = cv2.resize(gxy,None,fx=rs, fy=rs, interpolation = cv2.INTER_LANCZOS4)

    # Compute local orientations
    Ohat = np.zeros(gxx.shape)
    
    M = np.zeros([gxx.shape[0], gxx.shape[1], 2,2])
    M[:,:,0,0] = gxx
    M[:,:,1,1] = gyy
    M[:,:,0,1] = gxy
    M[:,:,1,0] = gxy
    D, V = np.linalg.eig(M)

    ind0 = (np.sign(D[:,:,0] - D[:,:,1])+1)/2
    ind1 = (np.sign(D[:,:,1] - D[:,:,0])+1)/2

    Ohat = np.arctan2(V[:,:,1,0]*ind0 + V[:,:,1,1]*ind1,V[:,:,0,0]*ind0 + V[:,:,0,1]*ind1)
    
    # Align the orientation such that 0 radians is north, pi/2 is west and pi is south
    Ohat = np.mod(-Ohat,np.pi)

    Chat = np.sqrt(gx**2 + gy**2)
    Chat = cv2.GaussianBlur(Chat,(ksize,ksize),sigma_c)
    if rs != 1:
        Chat = cv2.resize(Chat,None,fx=rs, fy=rs, interpolation = cv2.INTER_LANCZOS4)
    
    return Ohat, Chat


def TuningFunction(x,c_1,c_2,c_3,k1,m1,k2,m2):
    return c_1*np.tanh(c_2*x[0] + c_3)*np.exp(k1*np.cos(2*(x[1]-m1)) + k2*np.cos(4*(x[1]-m2)))


def MSE(x,y):
    return np.sum((x.flatten() - y.flatten())**2)/len(x.flatten())