import numpy as np

def DirichletSplit(labelarr, numclients: int, alpha: float, seed: int = 0):
    """
    Splits data among clients using a Dirichlet distribution to simulate
    non-IID scenarios.
    """
    randgen = np.random.RandomState(seed)
    labelarr = np.array(labelarr)
    numclasses = len(np.unique(labelarr))

    idxsbyclass = [np.where(labelarr == k)[0] for k in range(numclasses)]
    clientinds = [[] for _ in range(numclients)]

    for classidxs in idxsbyclass:
        randgen.shuffle(classidxs)

        propvec = randgen.dirichlet(alpha * np.ones(numclients))
        propvec = (len(classidxs) * propvec).astype(int)

        remcnt = len(classidxs) - propvec.sum()
        propvec[:remcnt] += 1

        startpos = 0
        for cli, cnt in enumerate(propvec):
            clientinds[cli].extend(classidxs[startpos:startpos + cnt])
            startpos += cnt

    for cli in range(numclients):
        randgen.shuffle(clientinds[cli])

    return clientinds
