# GUIDE model given an M x T matrix of summary statistics (`betas'), where M = number of genetic variants
# and T is the number of traits

def guide(betas, L=100):
    import numpy as np
    from sklearn.decomposition import FastICA
    M = max(betas.shape)
    
    betas_m = betas - np.mean(betas, axis = 0)
    betas_m = betas_m - np.mean(betas_m, axis = 1)[:,np.newaxis]   # mean centering the sum stats
    
    U, S, Vt = np.linalg.svd(betas_m, full_matrices=False)       # full SVD of G

    Uc = U[:, :L]
    Sc = S[:L]
    Vc = Vt[:L, :]
    UVc = np.concatenate((Uc, Vc.T)) / np.sqrt(2)
    
    ica = FastICA(max_iter=10000, tol=0.000001) #, random_state = 1)

    W_XL, W_LT_t = np.split(ica.fit_transform(UVc) * np.sqrt(2), [M])
    W_LT = W_LT_t.T
    mix = ica.mixing_
    
    return W_XL, W_LT, Sc, mix

# Computing the contribution scores given GUIDE weights
def contrib(W):         
    M = max(W.shape)
    K = min(W.shape)

    if W.shape[0] == M:     # ensure W is K x M
        W = W.T
    W_var = W**2

    W_var_sort = np.zeros((K,M))
    
    for i in range(K):
        W_var_sort[i,:] = np.sort(W_var[i,:])[::-1]
    W_var_cum=np.cumsum(W_var_sort,axis=1)/np.sum(W_var_sort,axis=1)[:,np.newaxis]

    return W_var_sort, W_var_cum

# Computing the genetic variance components given GUIDE weights
# Note: inputing W_XL computes the L->T variance components (and vice versa, with W_LT corresponding to X->L); see preprint for more details
def var_comp(betas,W):
    M = max(W.shape)
    if W.shape[1] == M:     # ensure W is M x K
        W = W.T
    
    if betas.shape[0] == M:
        G = betas.T @ W
    else:
        G = betas @ W
    var_comp = (G**2).T/np.sum(G**2,axis=1)
    var_comp = var_comp.T
    return var_comp
