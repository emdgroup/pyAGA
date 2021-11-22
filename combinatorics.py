def cycles(perm):
    """
    Calculate complete cycles in incomplete permutation with None entries, meaning
    return the cycles without these entries.
    :param perm:
    :return:
    """
    all_indices = set(range(len(perm)))
    cycles = []
    for i in range(len(perm)):
        if i in all_indices:
            cycle = [i]
            while True:
                all_indices.remove(i)
                i = perm[i]
                if i in cycle:
                    cycles.append(cycle)
                    break
                elif i is None or i not in all_indices:
                    break
                else:
                    cycle.append(i)
    return cycles


if __name__ == "__main__":
    perm = [1, 2, None, 4, 5, 3, 7, None, 6]
    print(cycles(perm))