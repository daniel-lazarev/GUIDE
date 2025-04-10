# GUIDE (Genomic Unmixing by Independent Decomposition)

Here we maintaining the code for GUIDE, including implementations of the main algorithm, associated functions (e.g., for estimating optimal model dimensionality or weight significance values), and scripts used for generating plots. See https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2 for more information. 

## Prerequisites:
- Python 3.6 or higher
- `numpy` (>= 1.18.5)
- `scipy` (>= 1.5.0)
- `scikit-learn` (>= 0.23.1)

## Operating Systems:
- Ubuntu 20.04+
- macOS 11+
- Windows 10+

## Hardware:
- Runs on standard desktop hardware.
- For large datasets, increased RAM (>8 GB) is recommended for performance.

## Installation
Clone the Repository:

```
git clone https://github.com/daniel-lazarev/GUIDE.git
```
Typical install time on a desktop computer <1min

Navigate to the Directory:

```
cd GUIDE
```

## Usage

### Function: `guide`
The `guide()` function builds a GUIDE model of the given summary statistics matrix (`betas`) using with latent dimensionality, `L`.

### Parameters:
- `betas` (numpy array): A matrix of summary statistics (e.g., beta coefficients, z-scores) of shape `(M, T)`, where `M` is the number of variants (e.g., genes, SNPs) and `T` is the number of traits.

- `L` (int, default=100): The number of latent factors.

- `mean_center` (bool, default=True): Whether to mean-center the data.

- `standardize` (bool, default=False): Whether to standardize the data (z-score normalization).

- `SE` (numpy array, optional): A matrix of standard errors with the same shape as `betas`. If provided, the data will be scaled element-wise by the corresponding standard error values.


### Returns:
The function returns four matrices:

- `W_XL`: variants-to-latents weight matrix (shape: M x L)

- `W_LT`: latents-to-traits weight matrix  (shape: T x L)

- `Sc`: Singular values corresponding to the components

- `mix`: ICA mixing matrix


### Example Usage:
```python
from GUIDE import guide, var_comp, contrib, logw_mat, guide_infer, entropy_plot
```
```python
# Run GUIDE with 50 components
W_XL, W_LT, Sc, mix = guide(betas, L=50)
```
Using the weights, we can compute the variance components and contributions scores to assess the loadings of latent factors on variants and on traits (as in Figures 2 and 3 in the paper).
To compute the variance components and contribution scores:
```python
var_comp_XL = var_comp(betas, W_LT)
var_comp_LT = var_comp(betas, W_XL)

contrib_XL = contrib(W_XL)
contrib_LT = contrib(W_LT)
```
Weight significance values (w-values) for the model weights are computed to quantify the significance of every weight and for downstream analyses using w-value thresholds.
```python
logw_mat_XL, logw_mat_TL = logw_mat(betas, W_XL, W_LT)
```

Given user-inputted weight significance thresholds for traits (`thr_T`), latent factors (`thr_L`), and variants (`thr_X`), `guide_infer` uses the computed weight significance values to give a list of significant traits for every latent, the indices of statistically significant traits, latents for those traits, and variants for those latents, as well as their corresponding `-log10(w)` values. This allows the user to automate the analyses using the given GUIDE model. 
```python
traits_per_lat, sig_trait_idx, sig_lat_idx, sig_vars_idx, sig_trait_logw, sig_lat_logw, sig_vars_logw = guide_infer(logw_mat_XL,logw_mat_TL, thr_T = 8, thr_L = 8, thr_X = 8)
```

Entropy plots, which, as detailed in the paper, can be used to determine the amount of information gained by a GUIDE model, as compared with a truncated SVD model, as a function of the number of latent factors can be generated as follows. Note that this is the most computationally expensive function in the package and is best used with smaller datasets.
```python
# Generating the values for the models
ent_XL_GUIDE, ent_LT_GUIDE, ent_XL_SVD, ent_LT_SVD, bad_L, bad_L_diff = entropy_plot(G, L_start=1, L_stop=100, step=1, metric='contrib')
```
The above generates models with a starting number of latent factors `L_start=1` and stops at `L_stop=100` latents, with a step size of `step=1` (which could be increased to reduce computational cost). 
```python
# Generating the plots
plt.plot(ent_LT_GUIDE,label="GUIDE")
plt.plot(ent_LT_SVD,label="TSVD")
plt.legend(loc='upper right')
plt.ylabel('Entropy',size=14)
plt.xlabel('Number of latent factors',size=14)
plt.title("Entropy, $\\mathcal{L} \\rightarrow \\mathcal{T}$, UKB",size=17)

#plt.savefig('entropy-LT___.png',dpi=400)
plt.show()

plt.plot(ent_XL_GUIDE,label="GUIDE")
plt.plot(ent_XL_SVD,label="TSVD")
plt.legend(loc='upper right')
plt.ylabel('Entropy',size=14)
plt.xlabel('Number of latent factors',size=14)
plt.title("Entropy, $\\mathcal{X} \\rightarrow \\mathcal{L}$, UKB",size=17)
plt.show()


plt.plot(np.array(ent_XL_SVD)-np.array(ent_XL_GUIDE))
plt.ylabel('Entropy',size=14)
plt.xlabel('Number of latent factors',size=14)
plt.title("Entropy difference, $\\mathcal{X} \\rightarrow \\mathcal{L}$, UKB",size=17)
plt.show()
```

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
