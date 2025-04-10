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
from GUIDE import guide, logw_mat, guide_infer
```
```python
# Run GUIDE with 50 components, mean centering, and standardization
W_XL, W_LT, Sc, mix = guide(betas, L=50, mean_center=True, standardize=True)
```
```python
# Compute the weight significance values (w-values) for the weights
logw_mat_XL, logw_mat_TL = logw_mat(betas, W_XL, W_LT)
```
```python
traits_per_lat, sig_trait_idx, sig_lat_idx, sig_vars_idx, sig_trait_logw, sig_lat_logw, sig_vars_logw = guide_infer(logw_mat_XL,logw_mat_TL, thr_T = 8, thr_L = 8, thr_X = 8)
```
Given user-inputted weight significance thresholds for traits (`thr_T`), latent factors (`thr_L`), and variants (`thr_X`), `guide_infer` uses the computed weight significance values to give a list of significant traits for every latent, the indices of statistically significant traits, latents for those traits, and variants for those latents, as well as their corresponding `-log10(w)` values. This allows the user to automate the analyses using the given GUIDE model.
Other functions in `GUIDE`, such as `var_comp`, `contrib`, `entropy_plot`, are also provided to reproduce analyses shown in the paper. 




### Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
