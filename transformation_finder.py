import numpy as np

def find_trafos(coeff, accuracy):
    # form set of all correlation coefficients
    coeff = coeff.round(accuracy)
    coeff_vals = set()
    for i in coeff:
        for j in i:
            coeff_vals.add(j)
    # coeff_perms: each entry contains the edges with same correlation
    coeff_perms = []
    for i in coeff_vals:
        s = np.argwhere(coeff == i)
        coeff_perms.append([j for j in s if j[0] <= j[1]])        
    # poss: entry i is set of all possible rho(i) based on coeff_perms
    n = np.shape(coeff)[0]
    poss = [set(range(n))]*n
    # this is where the algorithm starts
    return calculate_trafos(coeff_perms, poss, [])
    
def calculate_trafos(perms, poss, res):
    if(all(len(i) == 1 for i in poss)):
        res.append([s.pop() for s in poss])
        return res
    elif(any(len(i) == 0 for i in poss)):
        return res
        
    s = min(([i for i in poss if len(i) != 1]), key=len)
    x = poss.index(s)

    for y in s:
        new_poss = filter_perms(perms, list(poss), x, y)
        res = calculate_trafos(perms, new_poss, res)
    return res

def filter_perms(perms, poss, x, y):
    for i in perms:
        s = set()
        for k in i:
            if(k[0] == y):
                s.add(k[1])
            elif(k[1] == y):
                s.add(k[0])
            else:
                continue
        for j in i:    
            if(j[0] == x):
                z = j[1]
            elif(j[1] == x):
                z = j[0]
            else: 
                continue
            poss[z] = poss[z].intersection(s)
    return poss