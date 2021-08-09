import numpy as np

def shif_by_one(i):
    return [i[8], i[9], i[1], i[11], i[12], i[4], i[14], i[0], i[7], i[2], 
            i[3], i[10], i[5], i[6], i[13]]

nums = [[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]

banner_world = []

for i in nums:
    banner_world.append(i)
    banner_world.append(shif_by_one(i))
    banner_world.append(shif_by_one(shif_by_one(i)))

banner_coeff = np.matrix.round(np.corrcoef(np.transpose(banner_world)),3)

two_by_two_world = [[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]

two_by_two_coeff = np.matrix.round(np.corrcoef(np.transpose(two_by_two_world)),3)