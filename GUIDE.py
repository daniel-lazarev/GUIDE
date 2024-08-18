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
    import numpy as np
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
    import numpy as np
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

def logp_vals(data, W_XL, W_LT, top_L=3, iters=1000):    # compute -log10(p-values) for every trait given variance contributions of top_L latents
    import numpy as np
    import scipy.stats as ss

    L,T = W_LT.shape
    
    varcomp_LT_GUIDE = var_comp(data,W_XL)
    
    varcomp_LT_GUIDE_list = []
    for t in range(T):
            varcomp_LT = np.sum(np.sort(varcomp_LT_GUIDE[t])[::-1][:top_L])
            varcomp_LT_GUIDE_list.append(varcomp_LT)
    
    var_comp_iters = []
    
    for i in range(iters):
        rotate_i = ss.ortho_group.rvs(L)       # rotation (orthogonal) matrix 
        W_LT_var_comp_i = var_comp(data,W_XL@rotate_i)
        
        varcomp_LT_list = []
        for t in range(T):
            varcomp_LT = np.sum(np.sort(W_LT_var_comp_i[t])[::-1][:top_L])
            varcomp_LT_list.append(varcomp_LT)
        var_comp_iters.append(varcomp_LT_list)
    
    var_comp_iters = np.array(var_comp_iters)
    
    log10p_list = []
    for trait in range(T):
        mean_T = np.mean(var_comp_iters.T[trait])
        std_T = np.std(var_comp_iters.T[trait])
        ln_p = -ss.norm(mean_T, std_T).logsf(varcomp_LT_GUIDE_list[trait])
        log10_p = ln_p/np.log(10)
        log10p_list.append(log10_p)
        
    return log10p_list


def logp_vals_mat(data, W_XL, W_LT, iters=100):    # M x L and T x L matrices containing -log10(p-values) for every pair of variant/trait and latent
    import numpy as np
    import scipy.stats as ss
    M,L = W_XL.shape
    T = W_LT.shape[1]
    
    varcomp_LT_GUIDE = var_comp(data,W_XL)
    varcomp_XL_GUIDE = var_comp(data,W_LT)
    
    W_IXL_var = []       # iters x M x L tensor
    W_ITL_var = []       # iters x T x L tensor

    for i in range(iters):        # every iteration is a different possible decomposition
        rotate_i = ss.ortho_group.rvs(L)       # rotation (orthogonal) matrix 
        W_LT_var_comp_i = var_comp(data,W_XL@rotate_i)
        W_XL_var_comp_i = var_comp(data,rotate_i.T@W_LT)

        W_ITL_var.append(W_LT_var_comp_i)
        W_IXL_var.append(W_XL_var_comp_i)
   
    W_IXL_var = np.array(W_IXL_var)
    W_ITL_var = np.array(W_ITL_var)
    
    logp_mat_XL = []
    logp_mat_TL = []

    for m in range(M):
        log10p_list = []
        for l in range(L):
            mean = np.mean(W_IXL_var[:,m,l])
            std = np.std(W_IXL_var[:,m,l])
            ln_p = -ss.norm(mean, std).logsf(varcomp_XL_GUIDE[m,l])
            log10_p = ln_p/np.log(10)
            log10p_list.append(log10_p)        
        logp_mat_XL.append(log10p_list)
    logp_mat_XL = np.array(logp_mat_XL)
    
    for t in range(T):
        log10p_list = []
        for l in range(L):
            mean = np.mean(W_ITL_var[:,t,l])
            std = np.std(W_ITL_var[:,t,l])
            ln_p = -ss.norm(mean, std).logsf(varcomp_LT_GUIDE[t,l])
            log10_p = ln_p/np.log(10)
            log10p_list.append(log10_p)    
        logp_mat_TL.append(log10p_list)
    logp_mat_TL = np.array(logp_mat_TL)
    
    return logp_mat_XL, logp_mat_TL
