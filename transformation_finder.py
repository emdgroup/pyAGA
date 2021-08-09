import numpy as np

def find_trafos(coeff, accuracy):
    # form set of all correlation coefficients
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
        # this is used to create poss, taking number of occurances into account
        # alterantively, the next 7 lines can simply be replaced by:
        # poss_temp.append(set(s.flatten()))
        unique, counts = np.unique([j for j in s if j[0] <= j[1]], return_counts=True)
        for j in set(counts):
            t = set()
            for k in range(len(unique)):
                if counts[k] == j:
                    t.add(unique[k])
            poss_temp.append(t)
        
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