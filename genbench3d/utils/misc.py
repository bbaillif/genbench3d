import numpy as np
from typing import Iterable, Any

def get_full_matrix_from_tril(tril_matrix: Iterable[Any], 
                              n: int) -> np.ndarray:
    # let the desired full distance matrix between 4 samples be
    # [a b c d
    #  e f g h
    #  i j k l
    #  m n o p]
    # where the diagonal a = f = l = p = 0
    # having b = e, c = i, d = m, g = j, h = n, l = o by symmetry
    # scipy squareform works on triu [b c d g h l]
    # while we want to work from tril [e i j m n o]
    
    matrix = np.zeros((n, n))
    i=1
    j=0
    for v in tril_matrix:
        matrix[i, j] = matrix[j, i] = v
        j = j + 1
        if j == i:
            i = i + 1
            j = 0
    return matrix