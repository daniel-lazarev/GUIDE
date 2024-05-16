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
