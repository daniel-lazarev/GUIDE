# GUIDE model given an M x T matrix of summary statistics (`betas'), where M = number of genetic variants, 
# T = the number of traits, and L = number of latent factors

def guide(betas, L=100, mean_center = True, standardize = False, SE = None):
    import numpy as np
    from sklearn.decomposition import FastICA
    M = max(betas.shape)
    T = min(betas.shape)
    betas_m = betas

    if betas_m.shape[1] == M:     # ensure betas is M x T
        betas_m = betas_m.T
    
    if SE is not None:
        if SE.shape != betas_m.shape:
            raise ValueError("Shape of SE must match shape of betas.")
        betas_m = betas_m / SE
    
    if mean_center:
        betas_m = betas_m - np.mean(betas_m, axis=0)
        betas_m = betas_m - np.mean(betas_m, axis=1, keepdims=True)

    if standardize:
        betas_m = (betas_m - np.mean(betas_m)) / np.std(betas_m)
        
    U, S, Vt = np.linalg.svd(betas_m, full_matrices=False)       # full SVD of betas

    Uc = U[:, :L]
    Sc = S[:L]
    Vc = Vt[:L, :]
    UVc = np.concatenate((Uc, Vc.T)) * np.sqrt((M+T)/2) #/ np.sqrt(2)
    
    ica = FastICA(max_iter=10000, tol=0.000001)

    W_XL, W_LT_t = np.split(ica.fit_transform(UVc) /np.sqrt((M+T)/2) , [M])   # * np.sqrt(2)
    W_LT = W_LT_t.T
    mix = ica.mixing_
    
    return W_XL, W_LT, Sc, mix


# Computing the contribution scores given GUIDE weights
def contrib(W):  
    import numpy as np
    M = max(W.shape)
    K = min(W.shape)

    if W.shape[0] == M:     # ensure W is L x M
        W = W.T
    W_var = W**2

    W_var_sort = np.zeros((K,M))
    
    for i in range(K):
        W_var_sort[i,:] = np.sort(W_var[i,:])[::-1]
    W_var_cum=np.cumsum(W_var_sort,axis=1)/np.sum(W_var_sort,axis=1)[:,np.newaxis]

    return W_var_sort, W_var_cum

# Computing the genetic variance components given GUIDE weights
# Note: inputing W_XL computes the L->T variance components (and vice versa, with W_LT corresponding to X->L); see preprint for more details

def var_comp(betas, W):
    import numpy as np

    # Ensure W is M x L
    if W.shape[1] > W.shape[0]:
        W = W.T  # Transpose to make shape (M x L)

    # Match dimensions for matrix multiplication
    G = betas.T @ W if betas.shape[0] == W.shape[0] else betas @ W

    # Compute variance components
    G_squared = G ** 2
    row_sums = np.sum(G_squared, axis=1, keepdims=True)
    var_comp = G_squared / row_sums

    return var_comp



def logw_mat(data, W_XL, W_LT):  
    import numpy as np
    from scipy import stats

    M, L = W_XL.shape
    T = W_LT.shape[1]

    # Compute variance contributions
    varcomp_LT_GUIDE = var_comp(data, W_XL)   # Shape: T x L
    varcomp_XL_GUIDE = var_comp(data, W_LT)   # Shape: M x L

    # Compute the beta distribution survival function values
    beta_sf = lambda s: stats.beta(1/2, (L-1)/2).sf(s)
    
    # Vectorized survival function and log10 transformation for XL
    w_vals_XL = beta_sf(varcomp_XL_GUIDE)
    logw_mat_XL = -np.log10(np.clip(w_vals_XL, a_min=1e-300, a_max=None))

    # Vectorized survival function and log10 transformation for TL
    w_vals_TL = beta_sf(varcomp_LT_GUIDE)
    logw_mat_TL = -np.log10(np.clip(w_vals_TL, a_min=1e-300, a_max=None))

    return logw_mat_XL, logw_mat_TL



def idx_from_var(weight_value, weight_mat):                 # finds the index of a variant/trait given a certain weight/score value 
    import numpy as np
    return np.where(weight_mat == weight_value)[0][0]       # weight_mat is a matrix of weights/scores (usually MxL or LxT)


def idx_from_var_list(weight_list, weight_mat):            # returns indices for a list of weights/scores
    import numpy as np
    idx_list = []
    for i in range(len(weight_list)):
        idx = idx_from_var(weight_list[i], weight_mat)
        idx_list.append(idx)
    return idx_list


# automatic inference based on -logw values. Can also use the contribution scores or variance components as inputs instead of the logw matrices

def guide_infer(logw_mat_XL,logw_mat_TL, thr_T = 10, thr_L = 10, thr_X = 10): 
    import numpy as np

    L = min(logw_mat_TL.shape)

    sort_logw_TL = np.sort(logw_mat_TL,axis=1)
    max_logw_TL = sort_logw_TL.T[-1]                               # list of highest logw_TL value for every trait
    sig_trait_logw = [x for x in max_logw_TL if x>thr_T]                 # traits with significant p-values in given GUIDE model
    sig_trait_idx = idx_from_var_list(sig_trait_logw,logw_mat_TL)  # indices of those traits
    sig_logw_mat_TL = logw_mat_TL[sig_trait_idx,:]                 # sub-matrix of logw_TL for those traits only

    sig_lat_idx = []
    sig_vars_idx = []
    sig_lat_logw = []
    sig_vars_logw = []
    for i in range(sig_logw_mat_TL.shape[0]):
        sig_logw_vals = [x for x in sig_logw_mat_TL[i] if x>thr_L]
        sig_lat_logw.append(sig_logw_vals)
        idx_list = idx_from_var_list(sig_logw_vals,sig_logw_mat_TL[i])      # significant latents for every trait with significant p-vals
        sig_lat_idx.append( idx_list )
    
    for j in range(L):                                           # every significant variant for every latent
            sig_logw_vals2 = [x for x in logw_mat_XL[:,j] if x>thr_X]
            sig_vars_logw.append(sig_logw_vals2)
            idx_list2 = idx_from_var_list(sig_logw_vals2,logw_mat_XL[:,j])   # significant variants for every latent with signficant p-vals
            sig_vars_idx.append(idx_list2)

    
    # returns list of significant traits for each latent, as computed by guide_infer
    # list given for every latent, so empty list signifies no significant traits for that latent with the given significance threshold
    def traits_per_sig_lat(sig_lat_idx, sig_trait_idx,L=100):     
        traits_for_lat = []
        for lat in range(L):
            sig_trait_per_lat = []
            for i,x in enumerate(sig_lat_idx):
                for j in range(len(x)):
                    if x[j]==lat:
                        sig_trait_per_lat.append(sig_trait_idx[i])
            traits_for_lat.append(sig_trait_per_lat) 
        return traits_for_lat
        
    traits_per_lat = traits_per_sig_lat(sig_lat_idx, sig_trait_idx,L=L)
    
    return traits_per_lat, sig_trait_idx, sig_lat_idx, sig_vars_idx, sig_trait_logw, sig_lat_logw, sig_vars_logw   
    # returns list of significant traits for every latent, and the indices of statistically significant traits, latents for those traits, and 
    # variants for those latents, as well as their corresponding -log10(p) values



# plot the entropy of the weights for all models with num of latent factors between L_start and L_stop
def entropy_plot(betas, L_start=1, L_stop=100, step=1, metric='contrib'):      # L_start, L_stop = values of numbers of latent factors, metric can be 
                                                                       # 'contrib' or 'var_comp'
    import numpy as np
    from sklearn.decomposition import FastICA
    import scipy.stats as ss
    
    M = max(betas.shape)
    T = min(betas.shape)
    betas_m = betas
    if betas_m.shape[1] == M:     # ensure betas is M x T
        betas_m = betas_m.T  
    betas_m = betas - np.mean(betas, axis = 0)
    betas_m = betas_m - np.mean(betas_m, axis = 1)[:,np.newaxis]   # mean centering the sum stats

    U, S, Vt = np.linalg.svd(betas_m, full_matrices=False)       # full SVD of betas

    ent_XL_GUIDE =[]
    ent_XL_SVD = []
    ent_LT_GUIDE =[]
    ent_LT_SVD = []
    bad_L = []
    bad_L_diff = []
    
    if metric=='contrib':
       for i in range(L_start,L_stop+1, step):
            Uc_i = U[:, :i]
            Vc_i = Vt[:i, :]
            UVc_i = np.concatenate((Uc_i, Vc_i.T)) * np.sqrt((M+T)/2)
            ica = FastICA(max_iter=10000, tol=0.000001) #, random_state = 1)

            Xc_i, Yt_rec_i = np.split(ica.fit_transform(UVc_i) / np.sqrt((M+T)/2), [M])
            Yc_i = Yt_rec_i.T
        
            if np.allclose(i, np.sum(Xc_i.T@Xc_i)) == False:
                bad_L.append(i)
                bad_L_diff.append(i-np.sum(Xc_i.T@Xc_i))
            
            ent_XL_GUIDE.insert(i,sum(ss.entropy(Xc_i**2)))
            ent_XL_SVD.insert(i,sum(ss.entropy( Uc_i**2  )))

            ent_LT_GUIDE.insert(i,sum(ss.entropy(Yc_i.T**2)))
            ent_LT_SVD.insert(i,sum(ss.entropy( Vc_i.T**2  )))

    if metric=='var_comp':
        for i in range(L_start,L_stop+1, step):
            Uc_i = U[:, :i]
            Vc_i = Vt[:i, :]
            UVc_i = np.concatenate((Uc_i, Vc_i.T)) * np.sqrt((M+T)/2)
            ica = FastICA(max_iter=10000, tol=0.000001) #, random_state = 1)

            Xc_i, Yt_rec_i = np.split(ica.fit_transform(UVc_i) / np.sqrt((M+T)/2), [M])
            Yc_i = Yt_rec_i.T
            comp_XL = var_comp(betas_m,Yc_i)
            comp_LT = var_comp(betas_m,Xc_i)
            comp_XL_svd = var_comp(betas_m,Vc_i)
            comp_LT_svd = var_comp(betas_m,Uc_i)
            
            if np.allclose(i, np.sum(Xc_i.T@Xc_i)) == False:
                bad_L.append(i)
                bad_L_diff.append(i-np.sum(Xc_i.T@Xc_i))
            
            ent_XL_GUIDE.insert(i,sum(ss.entropy(comp_XL)))
            ent_XL_SVD.insert(i,sum(ss.entropy( comp_XL_svd  )))

            ent_LT_GUIDE.insert(i,sum(ss.entropy(comp_LT)))
            ent_LT_SVD.insert(i,sum(ss.entropy( comp_LT_svd  )))
  
    return ent_XL_GUIDE, ent_LT_GUIDE, ent_XL_SVD, ent_LT_SVD, bad_L, bad_L_diff


def wape(X_true,X_pred):       # weighted absolute percent error. X_true, X_true are vectors
    import numpy as np
    a = np.sum(np.abs(X_true - X_pred))
    b = np.sum(np.abs(X_true))
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=~np.isclose(b,np.zeros_like(b)))
    return c


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Functions below are not routinely used for analyses with GUIDE and are included in case they may be helpful in some applications

#Monte Carlo approach for computing the w-values. Included for comparison with the closed-form approach. 
# For applications, use logw_mat, which provides the closed-form solution
def logw_vals_MC(data, W_XL, W_LT, iters=100):    # M x L and T x L matrices containing -log10(w-values) for every pair of variant/trait and latent
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
    
    logw_mat_XL = []
    logw_mat_TL = []

    for m in range(M):
        log10w_list = []
        for l in range(L):
            mean = np.mean(W_IXL_var[:,m,l])
            std = np.std(W_IXL_var[:,m,l])
            ln_w = -ss.norm(mean, std).logsf(varcomp_XL_GUIDE[m,l])
            log10_w = ln_w/np.log(10)
            log10w_list.append(log10_w)        
        logw_mat_XL.append(log10w_list)
    logw_mat_XL = np.array(logw_mat_XL)
    
    for t in range(T):
        log10w_list = []
        for l in range(L):
            mean = np.mean(W_ITL_var[:,t,l])
            std = np.std(W_ITL_var[:,t,l])
            ln_w = -ss.norm(mean, std).logsf(varcomp_LT_GUIDE[t,l])
            log10_w = ln_w/np.log(10)
            log10w_list.append(log10_w)    
        logw_mat_TL.append(log10w_list)
    logw_mat_TL = np.array(logw_mat_TL)
    
    return logw_mat_XL, logw_mat_TL

# computes a w-value (using the Monte Carlo method vs. the closed form above) for every trait based on the contribution of the top_L latent factors 
def logw_vals_MC(data, W_XL, W_LT, top_L=3, iters=1000):    # compute -log10(w-values) for every trait given variance contributions of top_L latents
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
    
    log10w_list = []
    for trait in range(T):
        mean_T = np.mean(var_comp_iters.T[trait])
        std_T = np.std(var_comp_iters.T[trait])
        ln_w = -ss.norm(mean_T, std_T).logsf(varcomp_LT_GUIDE_list[trait])
        log10_w = ln_w/np.log(10)
        log10w_list.append(log10_w)
        
    return log10w_list
