---
title: "Lecture: Applied Bayesian Statistics Using Stan: Basics"
output: 
  learnr::tutorial:
    progressive: true
    allow_skip: true
    css: css/learnr-theme.css
runtime: shiny_prerendered
---



## Stan

### What is Stan?

In the words of the developers:

<blockquote>
"Stan is a state-of-the-art platform for statistical modeling and high-performance statistical computation. Thousands of users rely on Stan for statistical modeling, data analysis, and prediction in the social, biological, and physical sciences, engineering, and business.
  
Users specify log density functions in Stan's probabilistic programming language and get:
  
- full Bayesian statistical inference with MCMC sampling (NUTS, HMC)
- approximate Bayesian inference with variational inference (ADVI)
- penalized maximum likelihood estimation with optimization (L-BFGS)
  
</blockquote>

<div style="text-align: right"> 
  <sub><sup>
    Source: https://mc-stan.org/ 
  </sub></sup>
</div>


### Why Stan?

- Open-source software
- Fast and stable algorithms
- High flexibility with few limitations
- Extensive documentation
    - [User's Guide](https://mc-stan.org/docs/2_19/stan-users-guide/index.html)
    - [Language Reference Manual](https://mc-stan.org/docs/2_19/reference-manual/index.html)
    - [Language Functions Reference](https://mc-stan.org/docs/2_19/functions-reference/index.html)
- Highly transparent development process; see [Stan Development Repository on Github](https://github.com/stan-dev/stan)
- Very responsive [Development Team](https://mc-stan.org/about/team/)
- Large and active community in the [Stan Forums](https://discourse.mc-stan.org/) and [Stack OVerflow](https://stackoverflow.com/questions/tagged/stan)
- Increasing number of [case studies](https://mc-stan.org/users/documentation/case-studies.html), [tutorials](https://mc-stan.org/users/documentation/tutorials.html), [papers and textbooks](https://mc-stan.org/users/documentation/external.html)
- Compatibility with various editor for syntax highlighting, formatting, and checking (incl. [RStudio](https://www.rstudio.com/) and [Emacs](https://www.gnu.org/software/emacs/))

### Stan interfaces

- RStan (R)
- PyStan (Python)
- CmdStan (shell, command-line terminal)
- MatlabStan (MATLAB)
- Stan.jl (Julia)
- StataStan (Stata)
- MathematicaStan (Mathematica)
- ScalaStan (Scala)

### (Some) R packages

- [**rstan**](https://cran.r-project.org/package=rstan): General R Interface to Stan
- [**shinystan**](https://cran.r-project.org/package=shinystan): Interactive Visual and Numerical Diagnostics and Posterior Analysis for Bayesian Models
- [**bayesplot**](https://cran.r-project.org/web/packages/bayesplot/index.html): Plotting functions for posterior analysis, model checking, and MCMC diagnostics.
- [**brms**](https://cran.r-project.org/package=brms): Bayesian Regression Models using 'Stan', covering a growing number of model types
- [**rstanarm**](https://cran.r-project.org/package=rstanarm): Bayesian Applied Regression Modeling via Stan, with an emphasis on hierarchical/multilevel models
- [**edstan**](https://cran.r-project.org/package=edstan): Stan Models for Item Response Theory
- [**rstantools**](https://cran.r-project.org/package=rstantools): Tools for Developing R Packages Interfacing with 'Stan'

### Caveat: Reproducibility 

Under what conditions are estimates reproducible? See [Stan Reference Manual](https://mc-stan.org/docs/2_19/reference-manual/reproducibility-chapter.html), Section 19:

- Stan version
- Stan interface (RStan, PyStan, CmdStan) and version, plus version of interface language (R, Python, shell)
- versions of included libraries (Boost and Eigen)
- operating system version
- computer hardware including CPU, motherboard and memory
- C++ compiler, including version, compiler flags, and linked libraries
- same configuration of call to Stan, including random seed, chain ID, initialization and data

## Bayesian workflow

### A quick overview

#### The short version

1. **Specification**: Specify the full probability model
    - data
    - likelihood
    - priors
2. **Model Building**: Translate the model into code
3. **Validation**: Validate the model with fake data
4. **Fitting**: Fit the model to actual data
5. **Diagnosis**: Check generic and algorithm-specific diagnostics to assess convergence
6. Posterior Predictive Checks
7. Model Comparison

<div style="text-align: right"> 
  <sub><sup>
    Source: [Jim Savage (2016) A quick-start introduction to Stan for economists. A QuantEcon Notebook.](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/IntroToStan_basics_workflow.ipynb)
  </sub></sup>
</div>

#### The long version

<img src="images/workflow.png" width="90%" style="display: block; margin: auto;" />

<div style="text-align: right"> 
  <sub><sup>
    Source: [Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P. C., & Modrák, M. (2020). Bayesian workflow.](https://arxiv.org/abs/2011.01808)
  </sub></sup>
</div>


## Specification (linear model)

### Reminder: Equivalent notations

1. Scalar form: $$y_i = \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \epsilon_i \text{ for all } i=1,...,N$$
2. Row-vector form: $$y_i = \mathbf{x_i^{\prime}} \mathbf{\beta} + \epsilon_i  \text{ for all } i=1,...,N$$
3. Column-vector form: $$\mathbf{y} = \beta_1 \mathbf{x_{1}} + \beta_2 \mathbf{x_{2}} + \beta_3 \mathbf{x_{3}} \mathbf{\epsilon}$$
4. Matrix form: $$\mathbf{y = X \beta + \epsilon}$$
	
### Example: Linear model

Our knowledge of generalized linear models gives us *almost* everything we need!

### Probability model for the data

First, let's recap the three parts of every GLM in the context of the linear model:

* Family: $\mathbf{y} \sim \text{Normal}(\eta, \sigma)$
* (Inverse) link function: $\mathbf{y^{\ast}} = \text{id}(\eta) = \eta$
* Linear component: $\eta = \mathbf{X} \beta$

The *family* specifies the probability model (a.k.a. likelihood, data-generating process, or generative model) for the *data*: The fundamental assumption of the linear model is that every observation $y_i$ is a realization from a normal pdf with location parameter (mean) $\eta_i$ and constant scale parameter (variance) $\sigma^2$.

*Note:* Mimicking the convention in both R and Stan, we parameterize the normal distribution in terms of its mean and *standard deviation* (not variance)!

### Known and unknown quantities

* Parameters (unknown, random quantities):
    * $\beta$, the coefficient vector
    * $\sigma$, the scale parameter of the normal
    * $\eta$, the location parameter of the normal
* Data (known, fixed quantities):
    * $\mathbf{y}$, the outcome vector
    * $\mathbf{X}$, the design matrix
    * the dimensions of $\mathbf{y}_{N \times 1}$ and $\mathbf{X}_{N \times K}$
    * the dimensions of $\beta_{K \times 1}$, $\sigma$ (a scalar), and $\eta_{N \times 1}$
    
### Priors

What is still missing are prior distributions for the unknown quantities.

Here, we have quite some discretion. There are few rules we must adhere to:

- Our $\beta$'s have unconstrained support (though by far not all value ranges may be reasonable!)
- The scale parameter $\sigma$ cannot be negative

Here, we will opt for a convenience solution and specify weakly informative zero-mean normal priors for the $\beta$'s and a weakly informative half-Cauchy prior for $\sigma$:

- $\beta \sim \text{N}(0, 10)$
- $\sigma \sim \text{Cauchy}^{+}(0, 5)$

<img src="04-lec_files/figure-html/prior-viz-1.png" width="624" style="display: block; margin: auto;" />



## Model building

### Stan Program Blocks

1. Functions: Declare user written functions
2. **Data**: Declare all known quantities
3. Transformed Data: Transform declared data inputs (once)
4. **Parameters**: Declare all unknown quantities
5. Transformed Parameters: Transform declared parameters (each step, each iteration)
6. **Model**: Transform parameters, specify prior distributions and likelihoods
7. Generated Quantities (each iteration)

### Program Blocks

<img src="images/blocks.png" width="72%" style="display: block; margin: auto;" />

<br>
<div style="text-align: right">
  <sub><sup>
    Source: http://mlss2014.hiit.fi/mlss_files/2-stan.pdf
  </sub></sup>
</div>

### Script for a Stan program

*Writing scripts for Stan programs*

- Start with a blank script in your preferred code editor and save it as "lm.stan" .
- This will enable syntax highlighting, formatting, and checking in RStudio and Emacs.
- Alternatively, you can save your model as a single character string in R (with some drawbacks).

*Style guide*

- There is a [style guide](https://mc-stan.org/docs/2_26/stan-users-guide/stan-program-style-guide.html). Some recommendations:
    - consistency
    - lines should not exceed 80 characters
    - lowercase variables, words separated by underscores
    - like R: space around operators: `y ~ normal(...)`, `x = (1 + 2) * 3`
    - spaces after commas are optional: `y[m,n] ~ normal(0,1)` or `y[m, n] ~ normal(0, 1)`
- Always make sure to end your script with a blank line.
- You must use a delimiter to finish lines: `;`.
- `// this is a comment`


### Data block

Declare all known quantities, including data types, dimensions, and constraints: 

- $\mathbf{y}_{N \times 1}$
- $\mathbf{X}_{N \times K}$

<div class="tutorial-exercise" data-label="ex1-data1" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
data {
  int<lower=1> N; // num. observations
  ... declarations ...
}
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>


<div class="tutorial-exercise-support" data-label="ex1-data1-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
data {
  int<lower=1> N; // num. observations
  int<lower=1> K; // num. predictors
  matrix[N, K] x; // model matrix
  vector[N] y;    // outcome vector
}
```

</div>

### Parameters block

Declare unknown 'base' quantities, including storage types, dimensions, and constraints: 

- $\beta$, the coefficient vector
- $\sigma$, the scale parameter of the normal

<div class="tutorial-exercise" data-label="ex1-pars1" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
parameters {
  ... declarations ...
}
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>


<div class="tutorial-exercise-support" data-label="ex1-pars1-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
parameters {
  vector[K] beta;      // coef vector
  real<lower=0> sigma; // scale parameter
}
```

</div>

### Transformed parameters block

Declare and specify unknown transformed quantities, including storage types, dimensions, and constraints: 

- $\eta = \mathbf{X} \beta$, the linear prediction


<div class="tutorial-exercise" data-label="ex1-tpars1" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
transformed parameters {
  ... declarations ... statements ....
}
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>


<div class="tutorial-exercise-support" data-label="ex1-tpars1-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
transformed parameters {
  vector[N] eta;  // declare
  eta = x * beta; // assign
}
```

</div>

### Model block

Declare and specify local variables (optional) and specify sampling statements:

- $\beta_k \sim \text{Normal}(0, 10) \text{ for k = 1,...,K}$ 
- $\sigma \sim \text{Cauchy}^{+}(0, 5)$
- $\mathbf{y} \sim \text{Normal}(\eta, \sigma)$

<div class="tutorial-exercise" data-label="ex1-mod1" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
model {
  // priors
  ... statements ...
  
  // log-likelihood
  ... statements ...
}
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

<div class="tutorial-exercise-support" data-label="ex1-mod1-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
model {
  // priors
  target += normal_lpdf(beta | 0, 10);  // priors for beta
  target += cauchy_lpdf(sigma | 0, 5);  // prior for sigma
  
  // log-likelihood
  target += normal_lpdf(y | eta, sigma);// likelihood
}
```

</div>

### Writing Stan programs in R

- You can supply Stan programs as a character string in R
- Downsides:
    - No syntax highlighting, formatting, and checking
    - Must use double quotation marks `"` around the string to avoid that the [transposition operator](https://mc-stan.org/docs/2_26/functions-reference/transposition-operator.html) `'` breaks the string
- Upsides: Works with the interactive `learnr` tutorials in our workshop!

<div class="tutorial-exercise" data-label="ex1-full" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="30">

```text
# Save as character
lm_code <- 
"data {
  int<lower=1> N; // num. observations
  int<lower=1> K; // num. predictors
  matrix[N, K] x; // design matrix
  vector[N] y;    // outcome vector
}

parameters {
  vector[K] beta;      // coef vector
  real<lower=0> sigma; // scale parameter
}

transformed parameters {
  vector[N] eta;  // declare lin. pred.
  eta = x * beta; // assign lin. pred.
}

model {
  // priors
  target += normal_lpdf(beta | 0, 10);  // priors for beta
  target += cauchy_lpdf(sigma | 0, 5);  // prior for sigma
  
  // log-likelihood
  target += normal_lpdf(y | eta, sigma); // likelihood
}"

# Write to script
writeLines(lm_code, con = "lm.stan")
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

## Validation

### Simulate the data-generating process in R

<div class="tutorial-exercise" data-label="inf-sim1" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="20">

```text
# Set seed
set.seed(20210329)

# Simulate data
N <- 1000L                                # num. observations
K <- 5L                                   # num. predictors
x <- cbind(                               # design matrix
  rep(1, N), 
  matrix(rnorm(N * (K - 1)), N, (K - 1))
  )

# Set "true" parameters
beta <- rnorm(K, 0, 1)                    # coef. vector
sigma <- 2.5                              # scale parameter

# Get transformed parameters
eta <- x %*% beta                         # linear prediction

# Simulate outcome variable
y_sim <- rnorm(N, eta, sigma)             # simulated outcome
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

### Setup and compilation

<div class="tutorial-exercise" data-label="inf-setup" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
## Setup
library(rstan)
rstan_options(auto_write = TRUE)             # avoid recompilation of models
options(mc.cores = parallel::detectCores())  # parallelize across all CPUs

## Data as list
standat_val <- list(
  N = N,
  K = K,
  x = x,
  y = y_sim
)

## C++ Compilation
lm_mod <- rstan::stan_model(model_code = lm_code)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

### Estimation

<div class="tutorial-exercise" data-label="inf-sampl" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="17">

```text
lm_val <- rstan::sampling(
  lm_mod,                     # compiled model
  data = standat_val,             # data input
  algorithm = "NUTS",         # algorithm
  control = list(             # control arguments
    adapt_delta = .85),
  save_warmup = FALSE,        # discard warmup sims
  sample_file = NULL,         # no sample file
  diagnostic_file = NULL,     # no diagnostic file
  pars = c("beta", "sigma"),  # select parameters
  iter = 2000L,               # iter per chain
  warmup = 1000L,             # warmup period
  thin = 2L,                  # thinning factor
  chains = 2L,                # num. chains
  cores = 2L,                 # num. cores
  seed = 20210329)            # seed
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

### Output summary

*Reminder:* Here are the 'true' parameter values:

<div class="tutorial-exercise" data-label="inf-out1" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
true_pars <- c(beta, sigma)
names(true_pars) <- c(paste0("beta[", 1:5, "]"), "sigma")
true_pars
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

And here are the estimates from our model:

<div class="tutorial-exercise" data-label="inf-out2" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
lm_val
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

- When comparing these estimates, the question is, of course, how much deviation should have us worried.
- Deviations from a single validation run may be due to a circumstantial simulation of 'extreme' outcome values when mimicking the data generating process. 
- [Cook, Gelman, and Rubin (2006)](https://www.tandfonline.com/doi/abs/10.1198/106186006X136976) thus recommend running many replications of such validation simulations.
- They also provide a useful test statistic.


### A stanfit object

<div class="tutorial-exercise" data-label="inf-out3" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
str(lm_val)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

## Inference

For the sake of illustration, we use the replication data from Bischof and Wagner (2019), made available through the [American Journal of Political Science Dataverse](https://doi.org/10.7910/DVN/DZ1NFG). 

The original analysis uses Ordinary Least Squares estimation to gauge the effect of the assassination of the populist radical right politician Pim Fortuyn prior to the Dutch Parliamentary Election in 2002 on micro-level ideological polarization. 

The outcome variable contains squared distances of respondents' left-right self-placement to the pre-election median self-placement of all respondents. The main predictor is a binary indicator whether the interview was conducted before or after Fortuyn's assassination. 

### Getting actual data

<div class="tutorial-exercise" data-label="inf-dat" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="30">

```text
## Retrieve and manage data
bw_ajps19 <-
  read.table(
    paste0(
      "https://dataverse.harvard.edu/api/access/datafile/",
      ":persistentId?persistentId=doi:10.7910/DVN/DZ1NFG/LFX4A9"
    ),
    header = TRUE,
    stringsAsFactors = FALSE,
    sep = "\t",
    fill = TRUE
  ) %>% 
  select(wave, fortuyn, polarization) %>% ### select relevant variables
  subset(wave == 1) %>%                   ### subset to pre-election wave
  na.omit()                               ### drop incomplete rows

## Define data
x <- model.matrix(~ fortuyn, data = bw_ajps19)
y <- bw_ajps19$polarization
N <- nrow(x)
K <- ncol(x)

## Collect as list
standat_inf <- list(
  N = N,
  K = K,
  x = x,
  y = y)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

### Inference

<div class="tutorial-exercise" data-label="inf-real" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="17">

```text
lm_inf <- rstan::sampling(
  lm_mod,                     # compiled model
  data = standat_inf,             # data input
  algorithm = "NUTS",         # algorithm
  control = list(             # control arguments
    adapt_delta = .85),
  save_warmup = FALSE,        # discard warmup sims
  sample_file = NULL,         # no sample file
  diagnostic_file = NULL,     # no diagnostic file
  pars = c("beta", "sigma"),  # select parameters
  iter = 2000L,               # iter per chain
  warmup = 1000L,             # warmup period
  thin = 2L,                  # thinning factor
  chains = 2L,                # num. chains
  cores = 2L,                 # num. cores
  seed = 20210329)            # seed
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

### Posterior summaries

The original analysis reports point estimates (standard errors) of 1.644 (0.036) for the intercept and -0.112 (0.076) for the before-/after indicator.

How do our estimates compare?

#### Model summary

<div class="tutorial-exercise" data-label="inf-sum" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
print(lm_inf,
      pars = c("beta", "sigma"),
      digits_summary = 3L)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

#### Hypothesis testing


```r
# Extract posterior samples for beta[2]
beta2_posterior <- rstan::extract(lm_inf)$beta[, 2]

# Probability that beta[2] is greater than zero
mean(beta2_posterior > 0)
```

```
## [1] 0.063
```

## Convergence diagnostics

### Generic diagnostics: `Rhat` and `n_eff`

1. $\hat{R} < 1.1$: Potential scale reduction statistic (aka Gelman-Rubin convergence diagnostic) 
    - low values indicate that chains are stationary (convergence to target distribution within chains)
    - low values indicate that chains mix (convergence to same target distribution across chains)
1. $\frac{\mathtt{n_{eff}}}{\mathtt{n_{iter}}} > 0.001$: Effective sample  size
    - A small effective sample size indicates high autocorrelation within chains
    - This indicates that chains explore the posterior density very slowly and inefficiently

### Algorithm-specific diagnostics

In the words of the developers:

<blockquote>
"Hamiltonian Monte Carlo provides not only state-of-the-art sampling speed, it also provides state-of-the-art diagnostics. Unlike other algorithms, when Hamiltonian Monte Carlo fails it fails sufficiently spectacularly that we can easily identify the problems."
</blockquote>

<div style="text-align: right"> 
  <sub><sup>
    Source: https://github.com/stan-dev/stan/wiki/Stan-Best-Practices 
  </sub></sup>
</div>

- Divergent transitions after warmup (validity concern)
    - increase `adapt_delta` (target acceptance rate)
    - reparameterize/optimize your code
- Maximum treedepth exceeded (efficiency concern)
    - increase `max_treedepth`
- Diagnostics summary for `stanfit` object: `check_hmc_diagnostics(object)`
- For further information, see the [Guide to Stan's warnings](https://mc-stan.org/misc/warnings.html)

### Visual diagnostics using shinystan

<div class="tutorial-exercise" data-label="shiny" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
library(shinystan)
launch_shinystan(lm_inf)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

Additional Functionality:

- `generate_quantity()`: Add a new parameter as a function of one or two existing parameters
- `deploy_shinystan()`: Deploy a 'ShinyStan' app on [shinyapps.io](https://www.shinyapps.io/)


### Visual diagnostics using **bayesplot**

**bayesplot** offers a vast selection of visual diagnostics for `stanfit` objects:

- Diagnostics for No-U-Turn-Sampler (NUTS) 
    - Divergent transitions
    - Energy
    - Bayesian fraction of missing information
- Generic MCMC diagnostics
   - $\hat{R}$
   - $\mathtt{n_{eff}}$
   - Autocorrelation
   - Mixing (trace plots)

For full functionality, examples, and vignettes:

- [GitHub Examples](https://github.com/stan-dev/bayesplot)
- [CRAN Vignettes](https://cran.r-project.org/web/packages/bayesplot/vignettes/visual-mcmc-diagnostics.html)
- `available_mcmc()`function

### Example: Trace plot for sigma

<img src="04-lec_files/figure-html/bayesplot-1.png" width="50%" style="display: block; margin: auto;" />

## Computational problems

Suppose one or several of the following apply:

- Your algorithm-specific diagnostics throw warnings (that don't go away easily)
- Your convergence diagnostics indicate signs of non-convergence (and increasing the warm-up period doesn't help)
- Everything converges, but you get non-sensical estimates and predictions
- You are confident that your model will eventually converge to a well-behaved target distribution but it takes *forever* and even lengthy runs won't get you there (yet).

Then what? 

### Addressing computational problems

Gelman et al. (2020) have some answers:

### Check for model misspecification

- Is the probability model correctly specified?
- Are constraints set adequately?
- Do your priors allow for posterior density in regions where you'd expect it?
- Are your parameters statistically identified?

### When a complex model fails: Reduce complexity

- Suppose you fit a mixture model with two equations, a constrained probabilistic mixing parameter, and various random effects in each equation
- Fit each equation separately without the random effects
- Fit the mixture model without the random effects
- Fit each equation with random effects

### Be time-efficient

- Test the model on small sets of well-behaved simulated data
- Test the model using short runs

### [Efficiency tuning](https://mc-stan.org/docs/2_26/stan-users-guide/optimization-chapter.html)

- Vectorize: Use matrix multiplication instead of loops through matrix rows.
- Reparameterize. There are some common suggestions (more in the next session).
- Standardize data inputs. Increases the chances that your parameters will be on similar scales (which may speed up computation).
- *Alternatively:* Use weakly informative data-dependent priors.
- Make priors more informative (if defensible): Can prevent chains from wandering far off the target distribution.
- Parallelize: Markov chains are independent. Let them run at the same time (if your CPUs allow for it)!
preservec816e7d5f6ed3413
preserveb54acda33a5ebe04
preservea38afe8bd4ad36b7
preservebb14e4b44c44f00d
preserve9293b0e27db05e1e
preserve134b7d9175888e99
preserveb922455ec21a0904
preserve7167b47f3338a269
preservebd755e1cac1807fd
preserve17f4839311af4ce5
preserve827759b01bb5b0b9
preserve96e9475d2ab35253
preservec9a61b9186e38485
preservef09249bf783dd701
preservebdf101365e358b88
preserve01be2c4fc755ea86
preserve2e04a3d743b993b8
preservefc902f66b94ca977

<!--html_preserve-->
<script type="application/shiny-prerendered" data-context="dependencies">
{"type":"list","attributes":{},"value":[{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["header-attrs"]},{"type":"character","attributes":{},"value":["2.16"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/pandoc"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["header-attrs.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["jquery"]},{"type":"character","attributes":{},"value":["3.6.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/3.6.0"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquery-3.6.0.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquerylib"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.1.4"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["bootstrap"]},{"type":"character","attributes":{},"value":["3.3.5"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/bootstrap"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["viewport"]}},"value":[{"type":"character","attributes":{},"value":["width=device-width, initial-scale=1"]}]},{"type":"character","attributes":{},"value":["js/bootstrap.min.js","shim/html5shiv.min.js","shim/respond.min.js"]},{"type":"character","attributes":{},"value":["css/cerulean.min.css"]},{"type":"character","attributes":{},"value":["<style>h1 {font-size: 34px;}\n       h1.title {font-size: 38px;}\n       h2 {font-size: 30px;}\n       h3 {font-size: 24px;}\n       h4 {font-size: 18px;}\n       h5 {font-size: 16px;}\n       h6 {font-size: 12px;}\n       code {color: inherit; background-color: rgba(0, 0, 0, 0.04);}\n       pre:not([class]) { background-color: white }<\/style>"]},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["pagedtable"]},{"type":"character","attributes":{},"value":["1.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/pagedtable-1.1"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["js/pagedtable.js"]},{"type":"character","attributes":{},"value":["css/pagedtable.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["highlightjs"]},{"type":"character","attributes":{},"value":["9.12.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/highlightjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["highlight.js"]},{"type":"character","attributes":{},"value":["textmate.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial.js"]},{"type":"character","attributes":{},"value":["tutorial.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-autocompletion"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-autocompletion.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-diagnostics"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-diagnostics.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-format"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmarkdown/templates/tutorial/resources"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-format.js"]},{"type":"character","attributes":{},"value":["tutorial-format.css","rstudio-theme.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["jquery"]},{"type":"character","attributes":{},"value":["3.6.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/3.6.0"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquery-3.6.0.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquerylib"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.1.4"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["navigation"]},{"type":"character","attributes":{},"value":["1.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/navigation-1.1"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tabsets.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["highlightjs"]},{"type":"character","attributes":{},"value":["9.12.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/highlightjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["highlight.js"]},{"type":"character","attributes":{},"value":["default.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["jquery"]},{"type":"character","attributes":{},"value":["3.6.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/3.6.0"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquery-3.6.0.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquerylib"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.1.4"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["font-awesome"]},{"type":"character","attributes":{},"value":["5.1.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/fontawesome"]}]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["css/all.css","css/v4-shims.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["bootbox"]},{"type":"character","attributes":{},"value":["4.4.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/bootbox"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["bootbox.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["idb-keyvalue"]},{"type":"character","attributes":{},"value":["3.2.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/idb-keyval"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["idb-keyval-iife-compat.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[false]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial.js"]},{"type":"character","attributes":{},"value":["tutorial.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-autocompletion"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-autocompletion.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-diagnostics"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-diagnostics.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]}]}
</script>
<!--/html_preserve-->
<!--html_preserve-->
<script type="application/shiny-prerendered" data-context="execution_dependencies">
{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["packages"]}},"value":[{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["packages","version"]},"class":{"type":"character","attributes":{},"value":["data.frame"]},"row.names":{"type":"integer","attributes":{},"value":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108]}},"value":[{"type":"character","attributes":{},"value":["abind","backports","base","base64enc","bayesplot","bslib","cachem","callr","checkmate","cli","coda","codetools","colorspace","colourpicker","compiler","crayon","crosstalk","curl","datasets","digest","distributional","dplyr","DT","dygraphs","ellipsis","evaluate","fansi","farver","fastmap","generics","ggplot2","ggridges","glue","graphics","grDevices","grid","gridExtra","gtable","gtools","highr","htmltools","htmlwidgets","httpuv","igraph","inline","jquerylib","jsonlite","knitr","labeling","later","lattice","learnr","lifecycle","loo","magrittr","markdown","matrixStats","methods","mime","miniUI","munsell","parallel","pillar","pkgbuild","pkgconfig","plyr","posterior","prettyunits","processx","promises","ps","purrr","R6","Rcpp","RcppParallel","reshape2","rlang","rmarkdown","rprojroot","rstan","rstantools","rstudioapi","sass","scales","shiny","shinyjs","shinystan","shinythemes","StanHeaders","stats","stats4","stringi","stringr","tensorA","threejs","tibble","tidyselect","tools","utf8","utils","V8","vctrs","withr","xfun","xtable","xts","yaml","zoo"]},{"type":"character","attributes":{},"value":["1.4-5","1.4.1","4.2.1","0.1-3","1.9.0","0.4.0","1.0.6","3.7.2","2.1.0","3.4.0","0.19-4","0.2-18","2.0-3","1.1.1","4.2.1","1.5.1","1.2.0","4.3.2","4.2.1","0.6.29","0.3.1","1.0.10","0.25","1.1.1.6","0.3.2","0.16","1.0.3","2.1.1","1.1.0","0.1.3","3.3.6","0.5.4","1.6.2","4.2.1","4.2.1","4.2.1","2.3","0.3.1","3.9.3","0.9","0.5.3","1.5.4","1.6.6","1.3.5","0.3.19","0.1.4","1.8.0","1.40","0.4.2","1.3.0","0.20-45","0.10.1","1.0.2","2.5.1","2.0.3","1.1","0.62.0","4.2.1","0.12","0.1.1.1","0.5.0","4.2.1","1.8.1","1.3.1","2.0.3","1.8.7","1.3.1","1.1.1","3.7.0","1.2.0.1","1.7.1","0.3.4","2.5.1","1.0.9","5.1.5","1.4.4","1.0.5","2.16","2.0.3","2.26.13","2.2.0","0.14","0.4.2","1.2.1","1.7.2","2.1.0","2.6.0","1.2.0","2.26.13","4.2.1","4.2.1","1.7.8","1.4.1","0.36.2","0.3.3","3.1.8","1.1.2","4.2.1","1.2.2","4.2.1","4.2.1","0.4.1","2.5.0","0.33","1.8-4","0.12.1","2.3.5","1.8-11"]}]}]}
</script>
<!--/html_preserve-->
