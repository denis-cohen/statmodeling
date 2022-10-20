---
title: "Lab: Applied Bayesian Statistics Using Stan: Extensions"
output: 
  learnr::tutorial:
    progressive: true
    allow_skip: true
    css: css/learnr-theme.css
runtime: shiny_prerendered
---




## The hierarchical logit model

### Prompt

We will pick things up right where we left off.

We now want to apply the varying intercept, varying slope logit model to our running example of vote choice for the AfD.

Specifically, we want to know whether anti-immigration preferences predict voting for the AfD more strongly among respondents in Bundesländer where the immigrant population is large.

### An explicit multi-level notation

To make the multi-level structure of our model explicit, we can use a two-level notation for the linear predictor:

$$
\begin{split}
\eta_{ij} & = \beta_{1j} + \beta_{2j} \mathtt{la\_self}_i  \\
\beta_{1j} & = \beta_1 + \gamma_{1} \mathtt{bl\_immig}_j + \nu_{j1} \\
\beta_{2j} & = \beta_2 + \gamma_{2} \mathtt{bl\_immig}_j + \nu_{j2}
\end{split}
$$
The level-two equations show how we want to use the context-level variable $\mathtt{bl\_immig}_j$ to explain the variability in our varying intercepts and slopes.

Rearranging into a single equation yields

$$\begin{split}
\eta_{ij} & = \beta_1 + \gamma_{1} \mathtt{bl\_immig}_j + \nu_{j1} + (\beta_2 + \gamma_{2} \mathtt{bl\_immig}_j + \nu_{j2}) \mathtt{la\_self}_i + \epsilon_i \\
\eta_{ij} & = \beta_1 + \gamma_{1} \mathtt{bl\_immig}_j + \nu_{j1} + \beta_2 \mathtt{la\_self}_i + \gamma_{2} \mathtt{bl\_immig}_j \mathtt{la\_self}_i + \nu_{j2} \mathtt{la\_self}_i  + \epsilon_i \\
\eta_{ij} & = \beta_1 + \underbrace{\beta_2 \mathtt{la\_self}_i + \gamma_{1} \mathtt{bl\_immig}_j + \gamma_{2} \mathtt{bl\_immig}_j \mathtt{la\_self}_i}_{\text{interaction}} + \underbrace{\epsilon_i + \nu_{j1} + \nu_{j2} \mathtt{la\_self}_i}_{\text{composite errors}}
\end{split}$$

This is why we also call this a model with a *cross-level interaction*.

### An MLE reference model

Before we start, fit the same model using MLE estimation with `lme4::glmer()`.

Get a `summary()` of your model. Are there any irregularities you can spot?

<div class="tutorial-exercise" data-label="glmer" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0"><script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

<div class="tutorial-exercise-support" data-label="glmer-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
# Estimate
glmer_est <- lme4::glmer(
  vote_afd ~ la_self +
    bl_immig +
    bl_immig:la_self +
    (1 + la_self | bl),
  data = gles,
  family = binomial(link = "logit")
)

# Print summary
summary(glmer_est)
```

</div>

### Estimation problems

We can see (at least) two problems:

1. Convergence failure of MLE
1. Singularity: A correlation of $-1.00$ between intercept and slopes indicates that the model cannot adequately estimate the variance-covariance matrix of the random effects

This is not surprising!

  - We are running a complex model on a relatively small data set
  - The number of groups *and* the number of observations in some groups (Saarland!) are very low:


```r
table(gles$bl)
```

```
## 
##     Schleswig-Holstein                Hamburg          Niedersachsen 
##                     44                     28                    134 
##                 Bremen    Nordrhein-Westfalen                 Hessen 
##                     12                    206                     71 
##        Rheinland-Pfalz     Baden-Wuerttemberg                 Bayern 
##                     62                    141                    194 
##               Saarland                 Berlin            Brandenburg 
##                      5                     43                     98 
## Mecklenburg-Vorpommern                Sachsen         Sachsen-Anhalt 
##                     39                    113                     55 
##             Thueringen 
##                     76
```

This even more apparent when we consider the number of AfD voters per Bundesland (Bremen!):


```r
tapply(gles$vote_afd, gles$bl, sum)
```

```
##     Schleswig-Holstein                Hamburg          Niedersachsen 
##                      3                      2                      4 
##                 Bremen    Nordrhein-Westfalen                 Hessen 
##                      0                     12                      6 
##        Rheinland-Pfalz     Baden-Wuerttemberg                 Bayern 
##                      1                      8                      9 
##               Saarland                 Berlin            Brandenburg 
##                      1                      3                     18 
## Mecklenburg-Vorpommern                Sachsen         Sachsen-Anhalt 
##                      5                     16                      5 
##             Thueringen 
##                      7
```

### Compilation


```r
# Model code
rslogit_code <- "data {
  int<lower=1> N;                      // num. observations
  int<lower=1> K;                      // num. predictors w. varying coefs
  int<lower=1> L;                      // num. predictors w. fixed coefs
  int<lower=1> J;                      // num. groups
  array[N] int<lower=0, upper=J> jj;   // group index
  matrix[N, K] x;                      // matrix varying predictors, incl. intercept
  matrix[N, L] z;                      // matrix varying predictors, excl. intercept
  array[N] int<lower=0, upper=1> y;    // outcome
}

transformed data {
  row_vector[K] zeroes;
  zeroes = rep_row_vector(0.0, K);
}

parameters {
  // Fixed portions
  vector[K] beta;      // fixed coef predictors w. varying coefs
  vector[L] gamma;     // fixed coef predictors w. fixed coefs
  
  // Random portions
  vector[K] nu[J];                  // group-specific zero-mean deviations
  cholesky_factor_corr[K] L_Omega;  // prior correlation of deviations
  vector<lower=0>[K] tau;           // prior scale of deviations
}

transformed parameters {
  // Variance-covariance matrix of random deviatons
  matrix[K,K] Sigma;
  Sigma = diag_pre_multiply(tau, L_Omega) * diag_pre_multiply(tau, L_Omega)';
}

model {
  // local variables
  vector[N] eta;
  
  // linear predictor
  eta = x * beta + z * gamma;               // lin. pred. without random effect
  for (i in 1:N) {
    eta[i] = eta[i] + x[i, ] * nu[jj[i]];   // add group-specific random effects
  }
  
  // priors
  target += normal_lpdf(beta | 0, 2.5);             // priors for beta
  target += normal_lpdf(gamma | 0, 2.5);            // priors for gamma
  target += cauchy_lpdf(tau | 0, 2.5);               // prior for scales of nu
  target += lkj_corr_cholesky_lpdf(L_Omega | K);   // Cholesky LJK corr. prior
  target += multi_normal_lpdf(nu | zeroes, Sigma); // nu


  // log-likelihood
  target += bernoulli_logit_lpmf(y | eta); // likelihood
}"

# C++ Compilation
rslogit_mod <- rstan::stan_model(model_code = rslogit_code)
```


### Data

In order to specify the model in the form given above, we need two model matrices:

- $\mathbf{X}_{N\times2}$ accommodates a leading column of 1's and `la_self`
- $\mathbf{Z}_{N\times2}$ accommodates a `bl_immig` and the product term `bl_immig:la_self`

Generate these, and all other required inputs below:

<div class="tutorial-exercise" data-label="data" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
# Define data objects
y <- ...
x <- ...
z <- ...
z <- ...
jj <- ...
J <- ...
N <- ...
K <- ...
L <- ...

# Bind in a named list
standat_rslogit <- list(
  y = y,
  x = x,
  z = z,
  jj = jj,
  J = J,
  N = N,
  K = K,
  L = L
)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

<div class="tutorial-exercise-support" data-label="data-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
# Define data objects
y <- gles$vote_afd
x <- model.matrix(~la_self, data = gles)
z <- model.matrix(~bl_immig + bl_immig:la_self, data = gles)
z <- z[, 2:ncol(z)]
jj <- as.integer(gles$bl)
J <- length(unique(jj))
N <- nrow(x)
K <- ncol(x)
L <- ncol(z)

# Bind in a named list
standat_rslogit <- list(
  y = y,
  x = x,
  z = z,
  jj = jj,
  J = J,
  N = N,
  K = K,
  L = L
)
```

</div>

### Inference

We can fit the model using the following code:


```r
rslogit_est <- rstan::sampling(
  rslogit_mod,               
  data = standat_rslogit,         
  algorithm = "NUTS",
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.9
  ),
  pars = c("beta", "gamma", "Sigma", "nu"),  
  iter = 8000L,               
  warmup = 4000L,             
  thin = 4L,                  
  chains = 2L,                
  cores = 2L,                 
  seed = 20210330)   
```

As we know, the model is quite demanding. We are already doing quite a bit to ensure the validity of our estimates. In effort to avoid divergent transitions, non-convergence or poor mixing of chains, or low effective sample sizes due to high autocorrelation, we:

- run our chains for `iter = 8000L` iterations, of which we discard the first half as warmup.
- increase the algorithm control arguments `max_treedepth` and `adapt_delta`

The result is that the model shows no obvious signs of algorithm failure but takes quite some time to run (on my machine, approx. 20 minutes).

Therefore, we will not run it as part of this exercise; instead, you can directly assess the `stanfit` object in the next task!

### Model results and convergence diagnosis

Run the code to print the model summary. Do the generic diagnostics show any signs of non-convergence?

<div class="tutorial-exercise" data-label="print" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
print(rslogit_est, 
      pars = c("beta", "gamma", "Sigma"),
      digits = 4L)
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>


Use the [bayesplot](https://mc-stan.org/bayesplot/articles/visual-mcmc-diagnostics.html) package to try out some cool visual diagnostics.

<div class="tutorial-exercise" data-label="visualize" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="20"><script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

## Quantities of interest

### Conditional average marginal effects

We are now going to calculate our most challenging quantity of interest yet: The *conditional average marginal effect* of $\mathtt{la\_self}$ on  $\Pr(\mathtt{vote\_afd})$.

This quantity of interest is vastly similar to the average marginal effect from Lab 2. But since our predictor $\mathtt{la\_self}$ is interacted with the variable $\mathtt{bl\_immig}$, the marginal effect will change along the scale of $\mathtt{bl\_immig}$. The general intuition (in the context of OLS models) is explained by [Brambor, Clark, and Golder (2006)](https://www.jstor.org/stable/25791835#metadata_info_tab_contents).

The conditional average marginal effect for a one unit increase in $\mathtt{la\_self}$ at any given value of $\mathtt{bl\_immig}^{\ast}$ is defined as:

$$\frac{1}{N} \sum_{i=1}^{N} \left\{ \text{logit}^{-1}(\beta_1 + \beta_2 (\mathtt{la\_self}_i + 1) + \gamma_1 \mathtt{bl\_immig}^{\ast} + \gamma_2 (\mathtt{la\_self}_i + 1) \times \mathtt{bl\_immig}^{\ast}) \\ - \text{logit}^{-1}(\beta_1 + \beta_2 \mathtt{la\_self}_i  + \gamma_1 \mathtt{bl\_immig}^{\ast} + \gamma_2 \mathtt{la\_self}_i \times \mathtt{bl\_immig}^{\ast}) \right\}$$

Of course, we do not simply want the average marginal value conditional on a single value of  $\mathtt{bl\_immig}^{\ast}$ but on a *sequence of values* of $\mathtt{bl\_immig}^{\ast}$. This will allow us to see how the marginal effect changes along the values of the moderator.

### Calculate the QOI

<div class="tutorial-exercise" data-label="qoi" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="30">

```text
# Extract the model parameters
pars <- rstan::extract(rslogit_est)

# Extract the model matrices from standat_rslogit
X0 <- X1 <- standat_rslogit$x
Z0 <- Z1 <- standat_rslogit$z
X1[, "la_self"] <- X1[, "la_self"] + 1


# Define value sequence for bl_immig (approx range: 1-4)
bl_immig_vals <- seq(1, 4, 0.1)

# Initialize container (for posterior median, 2.5 and 97.5 percentiles)
came <- array(NA, dim = c(length(bl_immig_vals), 3L))

# Start loop
for (i in seq_along(bl_immig_vals)) {
  # Manipulate z matrices
  Z0[, c("bl_immig", "bl_immig:la_self")] <- ...
  Z1[, c("bl_immig", "bl_immig:la_self")] <- ...
  
  # Get unit-specific linear predictors
  eta0 <- ...
  eta1 <- .
  
  # Get unit-specific conditional marginal effects
  cme <- plogis(eta1) - plogis(eta0)
  
  # Average across observations, take quantiles
  ...
}

# Export newly created objects to global environment
export(environment())
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

<div class="tutorial-exercise-support" data-label="qoi-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
# Extract the model parameters
pars <- rstan::extract(rslogit_est)

# Extract the model matrices from standat_rslogit
X0 <- X1 <- standat_rslogit$x
Z0 <- Z1 <- standat_rslogit$z
X1[, "la_self"] <- X1[, "la_self"] + 1

# Define value sequence for bl_immig (approx range: 1-4)
bl_immig_vals <- seq(1, 4, 0.1)

# Initialize container (for posterior median, 2.5 and 97.5 percentiles)
came <- array(NA, dim = c(length(bl_immig_vals), 3L))

# Start loop
for (i in seq_along(bl_immig_vals)) {
  # Manipulate z matrices
  Z0[, c("bl_immig", "bl_immig:la_self")] <-
    cbind(bl_immig_vals[i], bl_immig_vals[i] * X0[, "la_self"])
  Z1[, c("bl_immig", "bl_immig:la_self")] <-
    cbind(bl_immig_vals[i], bl_immig_vals[i] * X1[, "la_self"])
  
  # Get unit-specific linear predictors
  eta0 <- X0 %*% t(pars$beta) + Z0 %*% t(pars$gamma)
  eta1 <- X1 %*% t(pars$beta) + Z1 %*% t(pars$gamma)
  
  # Get unit-specific conditional marginal effects
  cme <- plogis(eta1) - plogis(eta0)
  
  # Average across observations, take quantiles
  came[i, ] <- quantile(apply(cme, 2, mean), c(.5, .025, .975))
}

# Export newly created objects to global environment
export(environment())
```

</div>

### Visualization

<div class="tutorial-exercise" data-label="visualize2" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="65">

```text
# Plot
## Auxiliary objects
x_lines <-  seq(1, 4, .5)
y_lines <- seq(0, .125, .025)

## Canvas
plot(
  1,
  1,
  type = 'n',
  main = paste("Conditonal Average Marginal Efffect",
               sep = "\n"),
  axes = F,
  xlab = "Bundesland-level immigration rate",
  ylab = "AME on Pr(Vote AfD)",
  ylim = range(y_lines),
  xlim = range(x_lines)
)
axis(2, outer = FALSE)
axis(1)
rect(
  min(x_lines),
  min(y_lines),
  max(x_lines),
  max(y_lines),
  col = "gray95",
  border = NA
)
for (y in y_lines)
  segments(min(x_lines),
           y,
           max(x_lines),
           y,
           col = "white",
           lwd = 2)
for (x in x_lines)
  segments(x,
           min(y_lines),
           x,
           max(y_lines),
           col = "white",
           lwd = 2)

## Prediction
polygon(
  c(bl_immig_vals, rev(bl_immig_vals)),
  c(
    came[, 2],
    rev(came[, 3])
  ),
  col = adjustcolor("gray30", alpha.f = .2),
  border = NA
)
lines(bl_immig_vals,
      came[, 1],
      lty = 1,
      col = "gray10",
      lwd = 2)
lines(bl_immig_vals,
      came[, 2],
      lty = 1,
      col = "gray30")
lines(bl_immig_vals,
      came[ ,3],
      lty = 1,
      col = "gray30")
```

<script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

### Follow-up

Suppose you have included this plot in a submission. Reviewer 2 remarks that it is interesting to see that the average effect of a one-point increase in anti-immigration preferences on AfD voting more than doubles along the scale of state-level immigration rates. However, they point out that the credible intervals at the two ends of the $x$-axis widely overlap and, therefore, wonder if the reported effect modification is truly systematic.

Modify the code you used to compute the conditional average effect  to report the probability that the average effect of a one-point increase in anti-immigration preferences is greater when the state-level immigration rate is at its maximum as opposed to its minimum value.

<div class="tutorial-exercise" data-label="qoi2" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="36"><script type="application/json" data-opts-chunk="1">{"fig.width":6.5,"fig.height":4,"fig.retina":2,"fig.align":"default","fig.keep":"high","fig.show":"asis","out.width":624,"warning":true,"error":false,"message":true,"exercise.df_print":"paged","exercise.checker":"NULL"}</script></div>

<div class="tutorial-exercise-support" data-label="qoi2-solution" data-caption="Code" data-completion="1" data-diagnostics="1" data-startover="1" data-lines="0">

```text
# Extract the model parameters
pars <- rstan::extract(rslogit_est)

# Extract the model matrices from standat_rslogit
X0 <- X1 <- standat_rslogit$x
Z0 <- Z1 <- standat_rslogit$z
X1[, "la_self"] <- X1[, "la_self"] + 1

# Get min/max values of bl_immig
bl_immig_vals <- range(standat_rslogit$z[, 1])

# Initialize container (for full posterior)
came <- array(NA, dim = c(length(bl_immig_vals), nrow(pars$beta)))

# Start loop
for (i in seq_along(bl_immig_vals)) {
  # Manipulate z matrices
  Z0[, c("bl_immig", "bl_immig:la_self")] <-
    cbind(bl_immig_vals[i], bl_immig_vals[i] * X0[, "la_self"])
  Z1[, c("bl_immig", "bl_immig:la_self")] <-
    cbind(bl_immig_vals[i], bl_immig_vals[i] * X1[, "la_self"])
  
  # Get unit-specific linear predictors
  eta0 <- X0 %*% t(pars$beta) + Z0 %*% t(pars$gamma)
  eta1 <- X1 %*% t(pars$beta) + Z1 %*% t(pars$gamma)
  
  # Get unit-specific conditional marginal effects
  cme <- plogis(eta1) - plogis(eta0)
  
  # Average across observations
  came[i, ] <- apply(cme, 2, mean)
}

# Report probability of interest
mean(came[2, ] > came[1, ])
```

</div>
preserve762f327a8cd47b30
preservef015c117a6e0498d
preserve9b8cc936d502c77b
preserve48e3ba3d4d886319
preserve4245753d5cd0adfc
preserve3babb6aa3e16366e
preserve8bbfb12f6df8772f
preserve0fefb9dae85b55e4
preserve929dbc0e56af1c4d
preserve332b2466b659946f

<!--html_preserve-->
<script type="application/shiny-prerendered" data-context="dependencies">
{"type":"list","attributes":{},"value":[{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["header-attrs"]},{"type":"character","attributes":{},"value":["2.16"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/pandoc"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["header-attrs.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["jquery"]},{"type":"character","attributes":{},"value":["3.6.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/3.6.0"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquery-3.6.0.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquerylib"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.1.4"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["bootstrap"]},{"type":"character","attributes":{},"value":["3.3.5"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/bootstrap"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["viewport"]}},"value":[{"type":"character","attributes":{},"value":["width=device-width, initial-scale=1"]}]},{"type":"character","attributes":{},"value":["js/bootstrap.min.js","shim/html5shiv.min.js","shim/respond.min.js"]},{"type":"character","attributes":{},"value":["css/cerulean.min.css"]},{"type":"character","attributes":{},"value":["<style>h1 {font-size: 34px;}\n       h1.title {font-size: 38px;}\n       h2 {font-size: 30px;}\n       h3 {font-size: 24px;}\n       h4 {font-size: 18px;}\n       h5 {font-size: 16px;}\n       h6 {font-size: 12px;}\n       code {color: inherit; background-color: rgba(0, 0, 0, 0.04);}\n       pre:not([class]) { background-color: white }<\/style>"]},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["pagedtable"]},{"type":"character","attributes":{},"value":["1.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/pagedtable-1.1"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["js/pagedtable.js"]},{"type":"character","attributes":{},"value":["css/pagedtable.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["highlightjs"]},{"type":"character","attributes":{},"value":["9.12.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/highlightjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["highlight.js"]},{"type":"character","attributes":{},"value":["textmate.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial.js"]},{"type":"character","attributes":{},"value":["tutorial.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-autocompletion"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-autocompletion.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-diagnostics"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-diagnostics.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-format"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmarkdown/templates/tutorial/resources"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-format.js"]},{"type":"character","attributes":{},"value":["tutorial-format.css","rstudio-theme.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["jquery"]},{"type":"character","attributes":{},"value":["3.6.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/3.6.0"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquery-3.6.0.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquerylib"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.1.4"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["navigation"]},{"type":"character","attributes":{},"value":["1.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/navigation-1.1"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tabsets.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["highlightjs"]},{"type":"character","attributes":{},"value":["9.12.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/highlightjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["highlight.js"]},{"type":"character","attributes":{},"value":["default.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["jquery"]},{"type":"character","attributes":{},"value":["3.6.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/3.6.0"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquery-3.6.0.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["jquerylib"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.1.4"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["font-awesome"]},{"type":"character","attributes":{},"value":["5.1.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["rmd/h/fontawesome"]}]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["css/all.css","css/v4-shims.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["rmarkdown"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["2.16"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["bootbox"]},{"type":"character","attributes":{},"value":["4.4.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/bootbox"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["bootbox.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["idb-keyvalue"]},{"type":"character","attributes":{},"value":["3.2.0"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/idb-keyval"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["idb-keyval-iife-compat.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[false]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial.js"]},{"type":"character","attributes":{},"value":["tutorial.css"]},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-autocompletion"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-autocompletion.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["tutorial-diagnostics"]},{"type":"character","attributes":{},"value":["0.10.1"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/tutorial"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["tutorial-diagnostics.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["ace"]},{"type":"character","attributes":{},"value":["1.2.6"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/ace"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["ace.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["name","version","src","meta","script","stylesheet","head","attachment","package","all_files","pkgVersion"]},"class":{"type":"character","attributes":{},"value":["html_dependency"]}},"value":[{"type":"character","attributes":{},"value":["clipboardjs"]},{"type":"character","attributes":{},"value":["1.5.15"]},{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["file"]}},"value":[{"type":"character","attributes":{},"value":["lib/clipboardjs"]}]},{"type":"NULL"},{"type":"character","attributes":{},"value":["clipboard.min.js"]},{"type":"NULL"},{"type":"NULL"},{"type":"NULL"},{"type":"character","attributes":{},"value":["learnr"]},{"type":"logical","attributes":{},"value":[true]},{"type":"character","attributes":{},"value":["0.10.1"]}]}]}
</script>
<!--/html_preserve-->
<!--html_preserve-->
<script type="application/shiny-prerendered" data-context="execution_dependencies">
{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["packages"]}},"value":[{"type":"list","attributes":{"names":{"type":"character","attributes":{},"value":["packages","version"]},"class":{"type":"character","attributes":{},"value":["data.frame"]},"row.names":{"type":"integer","attributes":{},"value":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92]}},"value":[{"type":"character","attributes":{},"value":["base","bayesplot","boot","bslib","cachem","callr","cli","coda","codetools","colorspace","compiler","crayon","curl","datasets","digest","dplyr","ellipsis","evaluate","fansi","fastmap","foreign","generics","ggplot2","ggridges","glue","graphics","grDevices","grid","gridExtra","gtable","htmltools","htmlwidgets","httpuv","inline","jquerylib","jsonlite","knitr","later","lattice","learnr","lifecycle","lme4","loo","magrittr","markdown","MASS","Matrix","matrixStats","methods","mime","minqa","munsell","nlme","nloptr","parallel","pillar","pkgbuild","pkgconfig","prettyunits","processx","promises","ps","purrr","R6","Rcpp","RcppParallel","rlang","rmarkdown","rprojroot","rstan","rstantools","rstudioapi","sass","scales","shiny","splines","StanHeaders","stats","stats4","stringi","stringr","tibble","tidyselect","tools","utf8","utils","V8","vctrs","withr","xfun","xtable","yaml"]},{"type":"character","attributes":{},"value":["4.2.1","1.9.0","1.3-28","0.4.0","1.0.6","3.7.2","3.4.0","0.19-4","0.2-18","2.0-3","4.2.1","1.5.1","4.3.2","4.2.1","0.6.29","1.0.10","0.3.2","0.16","1.0.3","1.1.0","0.8-82","0.1.3","3.3.6","0.5.4","1.6.2","4.2.1","4.2.1","4.2.1","2.3","0.3.1","0.5.3","1.5.4","1.6.6","0.3.19","0.1.4","1.8.0","1.40","1.3.0","0.20-45","0.10.1","1.0.2","1.1-30","2.5.1","2.0.3","1.1","7.3-57","1.4-1","0.62.0","4.2.1","0.12","1.2.4","0.5.0","3.1-157","2.0.3","4.2.1","1.8.1","1.3.1","2.0.3","1.1.1","3.7.0","1.2.0.1","1.7.1","0.3.4","2.5.1","1.0.9","5.1.5","1.0.5","2.16","2.0.3","2.26.13","2.2.0","0.14","0.4.2","1.2.1","1.7.2","4.2.1","2.26.13","4.2.1","4.2.1","1.7.8","1.4.1","3.1.8","1.1.2","4.2.1","1.2.2","4.2.1","4.2.1","0.4.1","2.5.0","0.33","1.8-4","2.3.5"]}]}]}
</script>
<!--/html_preserve-->
