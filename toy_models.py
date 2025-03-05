import numpy as np
from sklearn.preprocessing import StandardScaler

def gen_rr(d, p, rank, coef):
    """
    Generating a reduced-rank matrix C of shape (d, p) and of rank 'rank' with values 
    uniformly sampled between 'coef' and 'coef' + 0.5.
    """
    # Generate random matrices A and B of appropriate dimensions
    A = np.random.uniform(coef, coef + 0.5, (d, rank))
    B = np.random.uniform(coef, coef + 0.5, (rank, p))
    # Compute the reduced-rank matrix C
    C = A @ B
    return C / np.sum(C)

def gen_data(intervention=2, n=300, d=10, p=10, r=1, l=10, rank=5, coef=1, confounding=False, noise='gaussian', a=1, b=0) :
    """
    Sampling (A, X, Y) for train and test distributions as proposed in the paper.
    
    Parameters:
        intervention (float): Scaling factor for intervention in A_test.
        n (int): Number of samples.
        d (int): Dimensionality of X_train and X_test.
        p (int): Dimensionality of Y_train and Y_test.
        r (int): Dimensionality of N_A_train, N_A_test.
        l (int): Dimensionality of A_train, A_test, H_train, H_test.
        rank (int): Rank of the reduced-rank matrix B_XY.
        coef (float): Lower bound for uniform sampling of matrix elements.
        
    Returns:
        A_train, X_train, Y_train, A_test, X_test, Y_test, B_XY, N_Y_train, N_Y_test
    """
    # Generate normal distributed matrices A_train and A_test
    A_train = np.random.normal(0, 1, size=(n, r))
    A_test = np.random.normal(0, intervention, size=(n, r))

    # Generate normal distributed matrices for noise and latent factors
    if noise == 'gaussian':
        N_A_train, N_X_train, N_Y_train, N_H_train = np.random.normal(b, a, (n, r)), np.random.normal(b, a, (n, d)), np.random.normal(b, a, (n, p)), np.random.normal(b, a, (n, l))
        N_A_test, N_X_test, N_Y_test, N_H_test = np.random.normal(b, a, (n, r)), np.random.normal(b, a, (n, d)), np.random.normal(b, a, (n, p)), np.random.normal(b, a, (n, l))
    if noise == 'uniform':
        N_A_train, N_X_train, N_Y_train, N_H_train = np.random.normal(a, b, (n, r)), np.random.normal(a, b, (n, d)), np.random.normal(a, b, (n, p)), np.random.normal(a, b, (n, l))
        N_A_test, N_X_test, N_Y_test, N_H_test = np.random.normal(a, b, (n, r)), np.random.normal(a, b, (n, d)), np.random.normal(a, b, (n, p)), np.random.normal(a, b, (n, l))
    elif noise == 'exponential' :
        N_A_train, N_X_train, N_Y_train, N_H_train = np.random.exponential(a, (n, r)), np.random.exponential(a, (n, d)), np.random.exponential(a, (n, p)), np.random.exponential(a, (n, l))
        N_A_test, N_X_test, N_Y_test, N_H_test = np.random.exponential(a, (n, r)), np.random.exponential(a, (n, d)), np.random.exponential(a, (n, p)), np.random.exponential(a, (n, l))
    elif noise == 'poisson' :
        N_A_train, N_X_train, N_Y_train, N_H_train = np.random.poisson(a, (n, r)), np.random.poisson(a, (n, d)), np.random.poisson(a, (n, p)), np.random.poisson(a, (n, l))
        N_A_test, N_X_test, N_Y_test, N_H_test = np.random.poisson(a, (n, r)), np.random.poisson(a, (n, d)), np.random.poisson(a, (n, p)), np.random.poisson(a, (n, l))
    
    # Regression weights sampling (as a reduced rank weight matrix)
    B_XY = gen_rr(d=d, p=p, rank=rank, coef=coef)
    
    # Generate training data
    H_train = N_H_train
    if confounding:
        H_train += A_train
    X_train = A_train + H_train + N_X_train
    Y_train = X_train @ B_XY + H_train + N_Y_train
    
    # Generate testing data
    H_test = N_H_test
    if confounding:
        H_test += A_test
    X_test = A_test + H_test + N_X_test
    Y_test = X_test @ B_XY + H_test + N_Y_test
    
    # Scaling 
    sc_A, sc_X, sc_Y = StandardScaler(), StandardScaler(), StandardScaler()
    # Fit and transform the training data
    A_train, X_train, Y_train, N_Y_train = sc_A.fit_transform(A_train), sc_X.fit_transform(X_train), sc_Y.fit_transform(Y_train), sc_Y.transform(N_Y_train)
    # Transform the testing data
    A_test, X_test, Y_test, N_Y_test = sc_A.transform(A_test), sc_X.transform(X_test), sc_Y.transform(Y_test), sc_Y.transform(N_Y_test)
    
    return A_train, X_train, Y_train, A_test, X_test, Y_test, B_XY, N_Y_train, N_Y_test
