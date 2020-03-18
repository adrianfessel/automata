# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:33:32 2020

@author: Adrian
"""

import numpy as np
import automata

import matplotlib.pyplot as plt

def empty_arena(size = (20, 20), rT = 5):

    """
    Generate field A with a gradient in direction (-1, 1). Allocate automata
    in the center in a circle in field M
    
    Input:
        size -- 2-tuple, field size
        rT -- radius of circle indicating the initial condition for the automata
        
    Output:
        A, R, M, S -- matrices of size size
        
    R and S are returned empty (np.zeros).
    
    """
    
    X, Y = np.meshgrid(np.arange(size[1]),np.arange(size[0]))
    r = np.sqrt((X-np.mean(X))**2+(Y-np.mean(Y))**2)
    A = automata.gen_gradient(size, (-1, 1))
    R, M, S = np.zeros(size), np.zeros(size), np.zeros(size)
    M[r <= rT] = 1
    
    return A, R, M, S

def utrap():

    size = (100, 300)

    A = automata.gen_gradient(size, (0, 1))
    
    R = np.zeros(size)
    R[15:85,245:250] = np.inf
    R[15:20,175:245] = np.inf
    R[80:85,175:245] = np.inf
      
    M = np.zeros(size)
    M[45:55, 10:20] = 1
    
    S = np.zeros(size)

    return A, R, M, S 
    
## p: Potencies, positive = repellent, negative = attractant
## larger values mean that the effect is stronger
pA = -1
pR = 1
pM = -1
pS = 1

## inverse Temperatures b = 1/kT, always positive. A smaller value means that the particle is more likely to act stochastically
bA = 1  # attractant
bR = 0.01   # repellent
bM = 10   # cohesion
bS = 1  # substance (repellent)

## parameters for substance creation and diffusion
diffS = 0.3 # diffusion constant (how fast does the substance diffuse?)
prodS = 0.005 # production rate (how much substance is produced per time step)

# A, R, M, S = empty_arena((50, 50), 4)
A, R, M, S = utrap()

P = {'A':pA, 'R':pR, 'M':pM, 'S':pS}
B = {'A':bA, 'R':bR, 'M':bM, 'S':bS}
F = {'A':A, 'R':R, 'M':M, 'S':S}

t = 0
T = 10000
dt = 0.01


while t < T:
    
    ci, co = automata.get_candidates(F['M'])
    
    if not automata.metropolis_step(ci, co, F, P, B):
        continue
    else:
        F['M'][ci], F['M'][co] = 0, 1 
        
        F['S'] = F['S'] + prodS*F['M']
        F['S'] = automata.diffuse(F['S'], diffS)
        
        t += dt
        
        if np.mod(t,50*dt) < dt:
        
            automata.plot(abs(P['A'])*F['A'], abs(P['R'])*F['R'], F['M'], abs(P['S'])*F['S'])
            
            plt.pause(0.000001)
            plt.draw()
            plt.cla()