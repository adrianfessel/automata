# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:44:15 2020

@author: Adrian
"""

import numpy as np
import networkx as nx

import automata, kuramoto

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

    """
    Generate field A with a gradient in direction (0, 1). Generate an 
    "infinitely-repelling" U-trap in field R, mimicking the situation in
    https://www.pnas.org/content/109/43/17490. Automata are initialized in
    field M.

    Input:
        None -- Size. trap geometry and automata placement are given explicitly
                below.
        
    Output:
        A, R, M, S -- matrices of size as specified below.
        
    S is returned empty (np.zeros).
    
    """
    
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
    
def gen_graph(M):
    
    """
    Generate 8-connectivity graph G from binary matrix M.
    
    Input:
        M -- binary matrix
        
    Output:
        G -- networkx graph
        I -- ID matrix, mapping node IDs to pixels
        
    Each foreground pixel in M is converted to a node of the graph G, connected
    to all neighboring (assuming 8-connectivity) foreground pixels. Pixel
    positions (y, x) are stored as node attributes 'pos'.
    
    """
    
    G = nx.Graph()
    pos = np.argwhere(M)
    
    for ID, posi in enumerate(pos):
        G.add_node(ID)
        G.nodes[ID]['pos'] = tuple(posi)
    
    for ID1 in G.nodes:
        for ID2 in G.nodes:
            
            if not ID1 == ID2 and not G.has_edge(ID1, ID2):
                d = np.sqrt((G.nodes[ID1]['pos'][0]-G.nodes[ID2]['pos'][0])**2 + (G.nodes[ID1]['pos'][1]-G.nodes[ID2]['pos'][1])**2)
                
                if d <= 1.1*np.sqrt(2):
                    G.add_edge(ID1, ID2)
    
    I = -1*np.ones(M.shape, dtype=np.int)
    
    for ID in G.nodes:
        I[G.nodes[ID]['pos']] = ID
    
    return G, I

def update_graph(G, I, ci, co, average_attributes=True):
    
    """
    Update graph G and ID matrix I if a node is moved from ci to co.
    
    Input: 
        G -- networkx graph
        I -- associated ID matrix
        ci, co -- coordinate pairs (y, x) of the node before and after moving
        average_attributes -- False: keep attributes
                              True: replace attributes by averaging over new
                                    neighbors
                                    
    Output:
        G -- updated networkx graph
        I -- updated ID matrix
        
    If average_attributes is set to True, the node attribute 'pos' is not
    averaged. 'pos' is updated to the new position in any case.    
    
    """
    
    IDi = I[ci]
    
    if average_attributes:
        attributes = [att for att in G.nodes[IDi] if not att=='pos']
    
    G.remove_node(IDi)
    G.add_node(IDi)
    G.nodes[IDi]['pos'] = co
    
    I[co] = IDi
    I[ci] = -1
    
    for IDn in G.nodes:
        if not IDn == IDi:
            d = np.sqrt((G.nodes[IDn]['pos'][0]-G.nodes[IDi]['pos'][0])**2 + (G.nodes[IDn]['pos'][1]-G.nodes[IDi]['pos'][1])**2)

            if d <= 1.1*np.sqrt(2):
                G.add_edge(IDn, IDi)
                
                nbrK = [G.edges[IDn,nbr]['K'] for nbr in G.neighbors(IDn) if not nbr==IDi]
                
                if nbrK:
                    G.edges[IDn,IDi]['K'] = np.mean(nbrK)
                else:
                    G.edges[IDn,IDi]['K'] = np.mean([G.edges[edge]['K'] for edge in G.edges if 'K' in G.edges[edge]])
                    
    
    if average_attributes:
        for att in attributes:
            G.nodes[IDi][att] = np.mean([G.nodes[IDn][att] for IDn in G.neighbors(IDi)])
    
    return G, I

def update_frequency(G, F, W, p=1, w=2*np.pi):
    
    """
    Update frequency based on attractants / repellents.
    
    Input:
        G -- networkx graph
        F -- dict of fields
        W -- dict of frequency responses (-1 = attractant, 1 = repellent)
        p -- responsiveness 
        w -- natural frequency
        
    Output:
        G -- updated graph
        
    If p == 0, the frequency remains unaltered.
    
    """
    
    if not p:
        return G
    
    else:
        
        chem = {}
        
        for k in W:

            chem[k] = {ID:F[k][G.nodes[ID]['pos']]/F[k].mean() if not F[k].mean() == 0 else 0 for ID in G.nodes}
            
        ars = {ID:np.sum([W[k]*chem[k][ID] for k in chem]) for ID in G.nodes}
        mars = np.max(np.abs([ars[ID] for ID in ars]))
        
        for ID in G.nodes:
            G.nodes[ID]['w'] = w*(1 - p + p*ars[ID]/mars)
    
        return G



pA = -1
pR = 1
pM = -1
pS = 1

bA = 1  # attractant
bR = 0.01   # repellent
bM = 10   # cohesion
bS = 1  # substance (repellent)

## parameters for substance creation and diffusion
diffS = 0#0.3 # diffusion constant (how fast does the substance diffuse?)
prodS = 0#0.005 # production rate (how much substance is produced per time step)

A, R, M, S = empty_arena((25, 25), 4)
# A, R, M, S = utrap()

## Potencies, positive = repellent, negative = attractant
## larger values mean that the effect is stronger
P = {'A':-1, 'R':1, 'M':-1, 'S':1}

## Inverse Temperatures b = 1/kT, always positive. A smaller value means that 
## the particle is more likely to act stochastically
B = {'A':1, 'R':0.01, 'M':10, 'S':1}

F = {'A':A, 'R':R, 'M':M, 'S':S}

## Frequency responses, positive = increase, negative = decrease
W = {'A':1, 'R':-1, 'S':-1}

# local coupling strength
KL = 1
# global coupling strength
KM = -0.1

# characteristic frequency
w0 = 2*np.pi
# frequency noise
dw = 0.1*w0

t = 0
T = 10000
dt = 0.1

G, I = gen_graph(M)
# assign starting phase at random and characteristic frequencies as specified
for ID in G.nodes:
    G.nodes[ID]['p'] = 2*np.pi*np.random.rand()
    G.nodes[ID]['w'] = w0 + dw*(2*np.random.rand()-1)
    
# assign local coupling (as implemented: equal weights)
for ID1, ID2 in G.edges:
    G.edges[ID1, ID2]['K'] = KL

import matplotlib.cm as cm
cmap = cm.get_cmap('hsv')

while t < T:
    
    ci, co = automata.get_candidates(F['M'])
    
    if not automata.metropolis_step(ci, co, F, P, B):
        continue
    else:
        F['M'][ci], F['M'][co] = 0, 1 
        
        F['S'] = F['S'] + prodS*F['M']
        F['S'] = automata.diffuse(F['S'], diffS)
        
        G, I = update_graph(G, I, ci, co)
        
        G = update_frequency(G, F, W, p = 0.5, w = w0)
        dp = kuramoto.kuramoto_step(G, KM, dw)
        
        for ID in G.nodes:
            G.nodes[ID]['p'] += dp[ID]*dt   
            
        t += dt
        
        if np.mod(t,1*dt) < dt:
        
            automata.plot(abs(P['A'])*F['A'], abs(P['R'])*F['R'], F['M'], abs(P['S'])*F['S'])
            
            p = np.asarray([G.nodes[i]['p'] for i in G.nodes])
            c = cmap(np.mod(p.flatten(),2*np.pi)/(2*np.pi))
            
            pos = {ID:np.asarray([G.nodes[ID]['pos'][1], G.nodes[ID]['pos'][0]]) for ID in G.nodes}
            
            nx.draw_networkx_nodes(G, pos, node_size=10, node_color=c)  
            nx.draw_networkx_edges(G, pos, edge_color='k')  
                
            plt.pause(0.000001)
            plt.draw()
            plt.cla()