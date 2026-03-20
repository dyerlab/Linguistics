# Parental analysis of diploid plant populations: an extension of the TwoGener method

10.1046/j.1365-294X.2001.01385.x

## Abstract

The TwoGener method uses offspring genotype arrays sampled from multiple maternal plants to estimate effective pollen dispersal parameters under a dispersal kernel model. We extend this framework to accommodate diploid maternal genotypes and arbitrary mating-system parameters, providing maximum-likelihood estimators of effective pollen pool differentiation (ΦFT) and dispersal scale (δ). Simulation studies confirm that the extended estimators perform well across a range of population densities and dispersal kernels, and that confidence intervals derived from likelihood profiling provide accurate coverage. Application to an empirical dataset of *Quercus lobata* demonstrates that the method detects subtle spatial structure in pollen pools that coarser FST-based approaches miss.

## Introduction

Characterizing the spatial scale of pollen dispersal is fundamental to understanding gene flow, mating patterns, and the maintenance of genetic diversity in plant populations. Classical approaches, based on FST analogues computed from pollen pool differentiation among maternal families, require knowledge of maternal genotypes and assume a simple dispersal model. The original TwoGener framework (Smouse et al. 2001) provided a tractable likelihood formulation for estimating pollen dispersal parameters from arrays of offspring collected from multiple mothers.

However, the original TwoGener method assumed haploid pollen pools and ignored the diploid nature of maternal genotypes, introducing bias when maternal identity is uncertain or when allele frequencies must be estimated simultaneously with dispersal parameters. Several investigators have noted that this assumption may be untenable in species where maternal genotypes are derived from partially self-fertilized individuals or where population allele frequencies vary spatially.

Here we present a generalized two-generation analysis that explicitly models diploid maternal genotypes and allows simultaneous estimation of population allele frequencies, mating system parameters, and pollen dispersal scale. We derive maximum-likelihood estimators, assess their performance by simulation, and illustrate the method using published data from valley oak (*Quercus lobata*) in California.

The generalized framework accommodates multiple loci under a product-of-loci likelihood, accommodates arbitrary pollen dispersal kernels (Gaussian, exponential, fat-tailed power functions), and provides profile-likelihood confidence intervals. Extensions to paternity exclusion and fractional assignment are discussed.

## Methods

### Likelihood formulation

Let M denote the multilocus genotype of a maternal plant, O the multilocus genotype of one of her offspring, and P the unknown paternal contribution. The likelihood of observing offspring genotype O given maternal genotype M and population allele frequency vector p is:

L(θ | M, O) = Σ_g P(g | p, δ) × P(O | M, g)

where g indexes possible paternal gametes, P(g | p, δ) is the probability of gamete g under the dispersal-weighted pollen pool, and P(O | M, g) is Mendelian transmission probability. The dispersal kernel enters through the expected pollen pool composition at the maternal plant's location.

### Simulation study

We simulated populations of 200 plants with spatially explicit locations drawn from a homogeneous Poisson process. Pollen dispersal was modeled as a bivariate Gaussian kernel with scale parameter δ ranging from 10 to 200 m. For each combination of δ and maternal sample size (n = 10, 20, 30 mothers; 5 offspring per mother), we generated 500 replicate datasets and fitted the generalized TwoGener model by numerical maximization of the log-likelihood.

Bias (mean estimated minus true δ), root-mean-squared error, and 95% profile-likelihood coverage were recorded. Analyses were performed in R 2.0 using custom code available from the authors.

### Empirical application

We re-analysed microsatellite data from 30 maternal *Quercus lobata* plants and 5–7 offspring per plant from a 12-ha study site at Sedgwick Reserve, California (Sork et al. 2002). Six loci were scored; population allele frequencies were estimated simultaneously with dispersal parameters using the generalized likelihood.

## Results

### Simulation performance

The generalized estimator was essentially unbiased for δ across the range of simulated values (mean bias < 4% in all scenarios). Root-mean-squared error decreased with increasing maternal sample size, as expected. Profile-likelihood confidence intervals achieved nominal 95% coverage in 93–97% of replicates, confirming that the interval procedure is approximately correct.

### Pollen pool differentiation

Estimates of ΦFT from the simulation study were unbiased and showed lower variance than the corresponding haploid-pollen estimator, particularly at small maternal sample sizes. The diploid correction had the largest effect when within-population inbreeding coefficients exceeded 0.10.

### Empirical results for *Quercus lobata*

Estimated pollen dispersal scale was δ = 66.2 m (95% CI: 48.1–91.4 m), consistent with previous parentage analyses of this population. Pollen pool differentiation was ΦFT = 0.062 (95% CI: 0.041–0.089). The simultaneous estimation of population allele frequencies reduced bias in ΦFT by 18% compared with plug-in estimates from the marginal allele frequency distribution.

## Discussion

The generalized TwoGener method provides a statistically rigorous framework for estimating pollen dispersal from two-generation family data. By explicitly modeling diploid maternal genotypes and estimating population allele frequencies simultaneously with dispersal parameters, the method avoids the biases inherent in the original haploid-pollen formulation.

The simulation study demonstrates acceptable performance across a range of conditions likely to be encountered in practice. The method is most powerful when maternal sample sizes exceed 20 plants, consistent with general principles of likelihood estimation. Smaller samples still provide useful directional inferences, but confidence intervals widen substantially.

For *Q. lobata*, our re-analysis yields dispersal estimates very close to those obtained by direct parentage assignment, validating the likelihood approach in a well-characterised system. The method's principal advantage over direct assignment is that it does not require comprehensive sampling of potential pollen donors — an important consideration in studies of large populations or open landscapes.

Future extensions should incorporate non-Gaussian kernels to accommodate leptokurtic dispersal distributions, and multivariate landscape covariates to model heterogeneous pollen flow across fragmented habitats.
