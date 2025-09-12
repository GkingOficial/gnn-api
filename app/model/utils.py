import numpy as np
import pandas as pd
import random
from rdkit import Chem
import tensorflow as tf
from sklearn.model_selection import KFold

def convert_to_graph(smiles_list, max_atoms):
    adj = []
    features = []
    atomlist=[]
    molList=[]
    count = 0
    for i in smiles_list:
        # Mol

        iMol = Chem.MolFromSmiles(i.strip())
        # print("this is i: ",i)

        if(iMol==None):
            count=count+1
            print(count)
            print(i)
            continue

        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        # print("this is iAdjTmp",iAdjTmp)
        # print(iAdjTmp.shape[0])

        if iAdjTmp.shape[0] > 170:
            print('Molécula i' + i)
            print(iAdjTmp.shape[0])
            input('iiiii')

        if( iAdjTmp.shape[0] <= max_atoms):
            # Feature-preprocessing
            iFeature = np.zeros((max_atoms, 61))
            iFeatureTmp = []

            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only

            
            iFeature[0:len(iFeatureTmp), 0:61] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            molList.append(iMol)
            # Adj-preprocessing
            iAdj = np.zeros((max_atoms, max_atoms))
            #print("this is iFeatureTmp",(len(iFeatureTmp)))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))

            # if i == 'o1cccc1':
            #     print("iAdj:\n")
            #     print(iAdj)
            
            adj.append(adj_k(np.asarray(iAdj), 1))
            #print("length of adj_k", len((np.asarray(iAdj))))
    features = np.asarray(features)
    adj = np.asarray(adj)

    # print(f'a_batch: {len(adj)}')
    # print(f'x_batch: {len(features)}')
    # input('dddd')
    
    #print("length of adj",len(adj))
    return adj, features

def adj_k(adj, k):
    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)
    #print("ret shape",len(ret))
    return convert_adj(ret)

def convert_adj(adj):

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d

def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',

                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',

                                       'V', 'Sb', 'Sn', 'Ag', 'Co', 'Se', 'Ti', 'Zn',

                                       'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Hg', 'Pb','Cl']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5])+

                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def np_sigmoid(x):
    return 1. / (1. + np.exp(-x))

def maxSum(arr, n, k):
    if (n < k):
        k = n

    res = 0
    start = 0
    end = k

    for i in range(k):
        if (arr[i] > 0):
            res += arr[i]
        else:
            if(end < n):
                start += 1
                end += 1
            else:
                if(k > 1):
                    k -= 1
                else:
                    start = 0
                    end = 0
                    return start, end
                
    curr_sum = res
    for i in range(k, n):
        if(arr[i] > 0):
            curr_sum += arr[i] - arr[i - k]
        if(curr_sum < res):
            res = curr_sum
            start = i - k + 1
            end = i + 1
            
    return list(range(start,end))
