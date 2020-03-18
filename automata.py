# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:07:42 2020

@author: Adrian
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import signal
from random import choice

def nannorm(X):
    
    """ 
    Minmax normalization of data containing nan values. 
    
    """
    
    Xmax, Xmin = np.nanmax(X), np.nanmin(X)
    
    return (X.copy()-Xmin)/(Xmax-Xmin)

def diffuse(X, sigma):
    
    """ 
    Simulate diffusion by convolution with a gaussian kernel.
    
    Input:
        X -- virtual concentration map before diffusion
        sigma -- gaussian kernel width
        
    Output:
        X -- virtual concentration map after diffusion
    
    """
    X = ndimage.gaussian_filter(X, sigma=sigma) #, mode='wrap')
    
    return X

def gen_gradient(size, direction):
    
    """
    Generate a linear, two-dimensional gradient.
    
    Input:
        size -- matrix shape, should be a 2-entry tuple (y, x)
        direction -- gradient direction, should be a 2-entry tuple (dy, dx)
    
    Output:
        G -- gradient matrix
        
    Example:    
        G = gen_gradient((Ny, Nx), (0, 1))
        returns a matrix of size (Ny, Nx) with a gradient increasing in x-direction
        
    Output will be normalized between 0, 1.
    
    """
    
    X, Y = np.meshgrid(np.arange(size[1]),np.arange(size[0]))
    
    G = direction[1] * X + direction[0] * Y
    
    G = G - np.min(G)
    G = G / np.max(G)
    
    return G

def plot(A, R, M, S):
    
    """
    Plot function for cellular automata in the ARMS-model.
    
    Input:
        A -- attractant field
        R -- repellent field
        M -- mold field (cellular automata)
        S -- substance field
    
    Color code:
        A -- red
        R -- green
        M -- yellow
        S -- blue
        
    np.inf values in R are shown in white.
    
    """
    
    rgb = np.zeros((*A.shape, 3), dtype=np.uint8)
    
    R[np.isinf(R)] = np.nan
    By, Bx = np.where(np.isnan(R))
    My, Mx = np.where(M > 0)
    
    # repellent, attractant, substance
    rgb[:,:,0] = nannorm(R)*255
    rgb[:,:,1] = nannorm(A)*255
    rgb[:,:,2] = nannorm(S)*255
    
    # mold
    rgb[My,Mx,0:2] = 255

    # barrier
    rgb[By,Bx,:] = 255
    
    plt.imshow(rgb)
    
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    
def get_candidates(B):
    
    """
    Function to propose pairs of candidate points.
    
    Input:
        B -- binary matrix, automata positions given as forground pixels
        
    Output:
        ci -- coordinate pair (y,x) within the foreground 
              but adjacent to the background
        co -- coordinate pair (y,x) within the background 
              but adjacent to the foreground
              
    Candidates are selected at random from the sets of pixels that satisfy:
        ci -- B[ci] = 1 and less than 7 foreground neighbors
        co -- B[co] = 0 and at least three foreground neighbors
    
    """
    
    K = np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
    S = signal.convolve2d(B != 0,K,mode='same') #, boundary='wrap')

    E = np.bitwise_and(B != 0, S < 7)
    D = np.bitwise_and(B == 0, S > 2)
    
    i, o = np.argwhere(E), np.argwhere(D)
    
    i = [(i[v,0], i[v,1]) for v in range(len(i))]
    o = [(o[v,0], o[v,1]) for v in range(len(o))]
    
    ci, co = choice(i), choice(o)
    
    return ci, co

def cohesion_energy(B,p,o=1):
    
    """
    Function to determine the number of foreground pixels in a neighborhood
    in relation to the neighborhood size.
    
    Input:
        B -- binary matrix
        p -- coordinate pair (y,x) indicating the pixel of interest
        o -- neighborhood size, o=1 corresponds to the 8-pixel neighborhood
    
    """
    
    B = B[p[0]-o:p[0]+o+1,p[1]-o:p[1]+o+1] != 0
    
    if np.min(B.shape) == 2*o + 1:
        B[o,o] = 0
        return np.sum(B)/(B.shape[0]*B.shape[1]-1)
    else:
        return 0
    
def metropolis_step(ci, co, F, P, B):
        
    """
    Perform a single iteration of the Metropolis algorithm to evaluate
    whether pixel ci is moved to the proposed location co.
    
    Input:
        ci, co -- coordinate pairs (y,x)
        F -- relevant fields, dict keyed by field name
        P -- field potencies, dict keyed by field name
        B -- inverse temperatures, dict keyed by field name
    
    Output:
        True/False -- indicate whether the step should be performed
    
    """
    
    dE = {}

    for k in F:
        
        if k == 'M':
            dE[k] = P[k] * (cohesion_energy(F[k],co,o=2) - cohesion_energy(F[k],ci,o=2))
        else:
            dE[k] = P[k] * (F[k][co]-F[k][ci])

    if np.sum([dE[k] for k in F]) <= 0:        
        return True
    elif np.random.rand() <= np.prod([np.min([1, np.exp(-B[k]*dE[k])]) for k in F]):        
        return True
    else:        
        return False
        