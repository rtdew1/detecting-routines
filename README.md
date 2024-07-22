# Detecting Routines: Implications for Ridesharing CRM

**Maintained by:** Ryan Dew (first.last@upenn.edu)

Code repository to accompany Detecting Routines: Implications for Ridesharing CRM.

Citation: Dew, R., Ascarza, E., Netzer, O., & Sicherman, N. (2024). Detecting Routines: Applications to Ridesharing Customer Relationship Management. Journal of Marketing Research, 61(2), 368-392. https://doi.org/10.1177/00222437231189185

The model code relies on python/PyMC/NumPyro. The results are analyzed using a mixture of Python and R. 

### Setup
1. This code was originally developed using Python version 3.10.x. The exact requirements are available in `requirements.txt`
2. IMPORTANT: The code was developed using `jaxlib` 3.15, which is not available on `pip`. To install it, you must point to a wheel supplied by JAX/Google. The full set of wheels is available here: https://storage.googleapis.com/jax-releases/nocuda/ Make sure your requirements.txt is edited to point to the wheel compatible with your system. The default link is to the Linux version.
3. Some analysis files are coded in R. They each have their own required libraries, listed at the top of the script. They were tested using R version 4.4.1 ("Race for Your Life")
4. Benchmark models may have their own requirements, and are included only for reference.


### To run the model
1. Ensure that the data files train.csv and test.csv exist in data/ 
   (see README.txt in data/ for more information)
2. Change to the model subdirectory: `cd code/model`
3. Use `runfile.sh` to run the model with your desired settings


### To analyze the results
1. Change to the analysis subdirectory: `cd code/analysis`
2. `process_results_and_compute_mae.ipynb` can be for an initial exploration of the results, including plotting some basic parameters, and computing the mean absolute error over the holdout data
3. For more advanced analyses, and more beautiful plots, use the R scripts. First run `process_model_outputs.R`, which generates a set of data objects that are saved in `analysis_inputs/`. Once those are saved, any of the other R files can be run.


## Changelog

### July 22, 2024, 4:55pm ET
* Completely revised repository to be more user-friendly, including fixing all broken filepaths, and ensuring availability of all scripts to generate intermediate results
* Now including script to generate simulated data
* Revised requirements.txt file.

### June 21, 2023, 3:30pm ET
* Added basic code
* Currently does *not* include simulated data
* Some paths may be broken because of refactoring for public release
