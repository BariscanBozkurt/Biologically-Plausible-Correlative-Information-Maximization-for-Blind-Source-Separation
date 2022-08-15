#######  GENERAL UTILITY FUNCTIONS ####################
import numpy as np
import itertools
import time
from IPython.display import display, Math

class Timer:
    """
    A simple timer class for performance profiling
    Taken from https://github.com/flatironinstitute/online_psp/blob/master/online_psp/online_psp_simulations.py
    
    Usage:
    with Timer() as t:
        DO SOMETHING HERE 
    print('Above (DO SOMETHING HERE) took %f sec.' % (t.interval))

    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

########### Set Theoretic Functions ##################
def findsubsets(S,m):
    """Find subsets of given set with given number of elements

    Args:
        S (set or list): A list of elements
        m (int): Number of elements of the subsets

    Returns:
        _type_ (set): All possible subsets of S with m elements

    Examples:
    >>> findsubsets({0,1,2,3}, 2)
    {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}

    >>> findsubsets([0,1,2,3], 3)
    {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}
    
    """
    return set(itertools.combinations(S, m))

def swapped(l, p1, p2):
    """Swap the elements of l in indices p1 and p2

    Args:
        l (list or numpy array): A list of elements whose elements to be swapped
        p1 (int): Swapping position 1
        p2 (int): Swappint position 2

    Returns:
        _type_ (list or numpy array): Resulting swapped list

    Examples: 
    >>> swapped([0,1,2,3,4], 1,3)
    [0, 3, 2, 1, 4]
    >>> swapped(np.array([0,1,2,3,4]), 1,2)
    array([0, 2, 1, 3, 4])
    """
    r = l[:]
    r[p1], r[p2] = r[p2], r[p1]
    return r

def transpositions(A):
    """Returns all transpositions of A

    Args:
        A (list or numpy array): 

    Returns:
        _type_ (list): List of all transpositions

    Examples:
    >>> transpositions([0,1,2])
    [[1, 0, 2], [2, 1, 0], [0, 2, 1]]
    >>> transpositions(np.array([0,1,2]))
    [array([2, 1, 0]), array([2, 1, 0]), array([2, 1, 0])]
    """
    return [swapped(A,i,j) for i,j in list(findsubsets(A,2))]

def elementary_perm_matrix(size,frm,to):
    """Returns an elementary permutation matrix (to permute from index 'frm' to index 'to')

    Args:
        size (int): Size of the permutation matrix
        frm (int): _description_
        to (int): _description_

    Returns:
        _type_ (numpy array): Elementary permutation matrix

    Examples:
    >>> elementary_perm_matrix(3,1,2)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    P = np.identity(size)
    P[:,[frm,to]] = P[:,[to,frm]]
    return P

def check_sign_perm(perm_mat):
    """Check if given matrix (perm_mat) is a signed permutation or not.

    Args:
        perm_mat (numpy array): Input (which will be checked to be signed permutation or not)

    Returns:
        _type_ (bool): Returns True if given matrix is signed permutation, and returns false if not.

    Examples: 
    >>> A = elementary_perm_matrix(3,1,2)
    >>> print(check_sign_perm(A))
    True
    >>> A2 = A.copy()
    >>> A2[0,1] = 1
    >>> print(check_sign_perm(A2))
    False
    """
    n = perm_mat.shape[0]
    if (np.linalg.norm((np.abs(perm_mat)>1e-6).sum(axis = 0) - np.ones((1,n))) < 1e-6):
        return True
    else: 
        return False
########### LATEX Style Display Matrix ###############
def display_matrix(array):
    """Display given numpy array with Latex format in Jupyter Notebook

    Args:
        array (numpy array): Array to be displayed
    """
    data = ''
    for line in array:
        if len(line) == 1:
            data += ' %.3f &' % line + r' \\\n'
            continue
        for element in line:
            data += ' %.3f &' % element
        data += r' \\' + '\n'
    display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))
