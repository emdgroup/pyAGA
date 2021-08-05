#import world_builder
import numpy as np
import pickle

with open("correlation_matrix_5x5_letters.pickle", "rb") as file:
    correlation_matrix = pickle.load(file)

def find_trafos(coeff, accuracy):
    # form set of all correlation coefficients (might have to be adjusted)
    coeff = coeff.round(accuracy)
    coeff_vals = set()
    for i in coeff:
        for j in i:
            coeff_vals.add(j)
    # coeff_perms: each entry contains the edges with same correlation
    # poss: entry i is set of all possible rho(i) based on coeff_perms
    coeff_perms = []
    poss_temp = []
    poss = []
    for i in coeff_vals:
        s = np.argwhere(coeff == i)
        coeff_perms.append([j for j in s if j[0] <= j[1]])
        poss_temp.append(set(s.flatten()))
        
    for i in range(np.shape(coeff)[0]):
        poss.append(set.intersection(*[j for j in poss_temp if i in j]))
    # this is where the algorithm starts
    return calculate_trafos(coeff_perms, poss, [])
    
def calculate_trafos(perms, poss, res):
    poss_min = len(min(poss, key=len))
    if(poss_min == 0):
        return res
    elif(poss_min == 1 and len(max(poss, key=len)) == 1):
        res.append(poss)
        return res
        

    s = min(([i for i in poss if len(i) != 1]), key=len)
    x = poss.index(s)
    
    if(s == set()):
        return
    for y in s:
        new_poss = filter_perms(perms, list(poss), x, y)
        res = calculate_trafos(perms, new_poss, res)
    return res

def filter_perms(perms, poss, x, y):
    for i in perms:
        for j in i:    
            if(j[0] == x):
                x2 = j[1]
            elif(j[1] == x):
                x2 = j[0]
            else: 
                continue
            s = set()
            for k in i:
                if(k[0] == y):
                    s.add(k[1])
                elif(k[1] == y):
                    s.add(k[0])
                else:
                    continue
            poss[x2] = poss[x2].intersection(s)
    return poss

#two_by_two_coeff = world_builder.two_by_two_coeff
print(find_trafos(correlation_matrix, 3))