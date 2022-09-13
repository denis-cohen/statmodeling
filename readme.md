
# Advanced Bayesian Statistical Modeling in R and Stan

Denis Cohen  
<denis.cohen@uni-mannheim.de>

## Abstract

Statistical models are widely used in the social sciences for
measurement, prediction, and hypothesis testing. While popular
statistical software packages cover a growing number of pre-implemented
model types, the diversification of substantive research domains and the
increasing complexity of data structures drive persistently high demand
for custom modeling solutions. Implementing such custom solutions
requires that researchers build their own models and use them to obtain
reliable estimates of quantities of substantive interest. Bayesian
methods offer a powerful and versatile infrastructure for these tasks.
Yet, seemingly high entry costs still deter many social scientists from
fully embracing Bayesian methods.

This workshop offers an advanced introduction to Bayesian statistical
modeling to push past these initial hurdles and equip participants with
the required skills for custom statistical modeling. Following a
targeted review of the underlying mechanics of generalized linear models
and core concepts of Bayesian inference, the course introduces
participants to Stan, a platform for statistical modelling and Bayesian
statistical inference. Participants will get an overview of the
programming language, the R interface RStan, and the workflow for
Bayesian model building, inference, and convergence diagnosis. Applied
exercises allow participants to write and run various model types and to
process the resulting estimates into publication-ready graphs.

## Prerequisites

Working knowledge of the software environment `R` as well as working
knowledge of (generalized) linear models is required for participation
in this course. Basic knowledge of linear algebra and probability theory
is recommended.

This workshop requires installations of recent versions of
[`R`](https://cran.r-project.org/mirrors.html) and
[`RStudio`](https://rstudio.com/products/rstudio/download/#download). On
In the second half of the workshop, we will use
[`Stan`](https://mc-stan.org/) via its R interface `RStan`. Setting up
`RStan` can be somewhat time-consuming as it requires the installation
of a C++ compiler. Workshop participants should follow [these
instructions](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started)
on the Stan Development Team’s GitHub to install and configure the
[`rstan`](https://cran.r-project.org/web/packages/rstan/index.html)
package and its prerequisites on their operating system *before the
workshop*. Should you encounter problems, feel free to send me an email.

# Course Structure

The workshop consists of five sessions à 180 minutes. Each session
starts with a lecture-style input talk, followed by lab-style applied
exercises.

| Session | Topics                                                                                                                                                                                                                                   |
|:-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    1    | **R Math & Programming Refresher**                                                                                                                                                                                                       |
|         | ***Session contents:***                                                                                                                                                                                                                  |
|         | \- Data types                                                                                                                                                                                                                            |
|         | \- Object types and conversions; slicing and indexing                                                                                                                                                                                    |
|         | \- Probability distributions                                                                                                                                                                                                             |
|         | \- Linear algebra                                                                                                                                                                                                                        |
|         | \- Control structures                                                                                                                                                                                                                    |
|         | \- Programming                                                                                                                                                                                                                           |
|         | ***Suggested readings:***                                                                                                                                                                                                                |
|         | \- Wickham, H. (2019). Advanced R. CRC press. Available online: <https://adv-r.hadley.nz/>                                                                                                                                               |
|         | \- Gill, J. (2006). Essential Mathematics for Political and Social Research. Cambridge: Cambridge University Press.                                                                                                                      |
|    2    | **Generalized Linear Models & Simulation-based Approaches to Inferential Uncertainty**                                                                                                                                                   |
|         | ***Session contents:***                                                                                                                                                                                                                  |
|         | \- GLM basics: Systematic component, link function, likelihood function                                                                                                                                                                  |
|         | \- GLM typology                                                                                                                                                                                                                          |
|         | \- The simulation approach: Quasi-Bayesian Monte Carlo simulation of model parameters                                                                                                                                                    |
|         | \- Quantities of interest (definition, calculation, simulation)                                                                                                                                                                          |
|         | ***Suggested readings:***                                                                                                                                                                                                                |
|         | \- Gill, J., & Torres, M. (2019). Generalized Linear Models: A Unified Approach (Vol. 134). SAGE Publications.                                                                                                                           |
|         | \- King, G., Tomz, M., & Wittenberg, J. (2000). Making the Most of Statistical Analyses: Improving Interpretation and Presentation. American Journal of Political Science, 44(2), 341–355.                                               |
|         | \- Gelman, A., & Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge: Cambridge University Press.                                                                                              |
|    3    | **Bayesian Fundamentals**                                                                                                                                                                                                                |
|         | ***Session contents:***                                                                                                                                                                                                                  |
|         | \- Fundamental concepts: Prior distribution, likelihood, posterior distribution                                                                                                                                                          |
|         | \- Analytical Bayes                                                                                                                                                                                                                      |
|         | \- MCMC Algorithms                                                                                                                                                                                                                       |
|         | \- Gibbs sampler implementation                                                                                                                                                                                                          |
|         | \- Convergence diagnostics                                                                                                                                                                                                               |
|         | ***Suggested readings:***                                                                                                                                                                                                                |
|         | \- Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2014). Bayesian data analysis (Vol. 2).                                                                                                                                      |
|         | \- Gill, J. (2015). Bayesian Methods. A Social and Behavioral Sciences Approach (3rd ed.). Boca Raton, FL: CRC Press.                                                                                                                    |
|         | \- Gill, J., & Heuberger, S. (2020). Bayesian Modeling and Inference: A Postmodern Perspective. In L. Curini & R. Franzese (Eds.), The SAGE Handbook of Research Methods in Political Science and International Relations (pp. 961–984). |
|    4    | **Applied Bayesian Statistics Using Stan: Basics & Workflow**                                                                                                                                                                            |
|         | ***Session contents:***                                                                                                                                                                                                                  |
|         | \- Stan: Language and documentation                                                                                                                                                                                                      |
|         | \- Core program blocks: Data, parameters, model                                                                                                                                                                                          |
|         | \- The Bayesian workflow I: Model specification, model building, validation, fitting, diagnosis                                                                                                                                          |
|         | \- Linear model implementation                                                                                                                                                                                                           |
|         | ***Suggested readings:***                                                                                                                                                                                                                |
|         | \- Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., … Modrák, M. (2020). Bayesian workflow. ArXiv. Available online: <https://arxiv.org/abs/2011.01808>                                                  |
|         | \- Nicenboim, B., Schad, D., & Vasishth, S. (2022). An Introduction to Bayesian Data Analysis for Cognitive Science. Retrieved from <https://vasishth.github.io/bayescogsci/book/>                                                       |
|         | \- Stan Documentation. Available online: <https://mc-stan.org/users/documentation/>                                                                                                                                                      |
|    5    | **Applied Bayesian Statistics Using Stan: Extensions and Advanced Modeling**                                                                                                                                                             |
|         | ***Session contents:***                                                                                                                                                                                                                  |
|         | \- Optional program blocks: Functions, transformed data, transformed parameters, generated quantities                                                                                                                                    |
|         | \- Efficiency tuning: Data pre-processing, reparameterization, NUTS control arguments                                                                                                                                                    |
|         | \- Hierarchical logistic model implementation                                                                                                                                                                                            |
|         | \- The Bayesian workflow II: Posterior predictive checks, model comparison                                                                                                                                                               |
|         | \- Processing posterior draws in Stan and R into substantively meaningful quantities                                                                                                                                                     |
|         | ***Suggested readings:***                                                                                                                                                                                                                |
|         | \- Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., … Modrák, M. (2020). Bayesian workflow. ArXiv. Available online: <https://arxiv.org/abs/2011.01808>                                                  |
|         | \- Nicenboim, B., Schad, D., & Vasishth, S. (2022). An Introduction to Bayesian Data Analysis for Cognitive Science. Retrieved from <https://vasishth.github.io/bayescogsci/book/>                                                       |
|         | \- Stan Documentation. Available online: <https://mc-stan.org/users/documentation/>                                                                                                                                                      |

## Using the workshop materials

The workshop materials come as
[`learnr`](https://rstudio.github.io/learnr/) tutorials wrapped in an R
package. To download, install, and use the interactive materials, run
the following code:

``` r
## Detach if loaded
if ("statmodeling" %in% (.packages())) {
  detach(package:statmodeling, unload = TRUE)
}

# Uninstall if installed
if ("statmodeling" %in% installed.packages()) {
  remove.packages("statmodeling")
}

# Install if not installed
if (!("devtools" %in% installed.packages())) {
  install.packages("devtools")
}

# Load from GitHub
library(devtools)
devtools::install_github("denis-cohen/statmodeling")

# Load to library
library(statmodeling)

# Run tutorials
learnr::run_tutorial("00-int", package = "statmodeling")
learnr::run_tutorial("01-lec", package = "statmodeling")
learnr::run_tutorial("01-lab", package = "statmodeling")
learnr::run_tutorial("02-lec", package = "statmodeling")
learnr::run_tutorial("02-lab", package = "statmodeling")
learnr::run_tutorial("03-lec", package = "statmodeling")
learnr::run_tutorial("03-lab", package = "statmodeling")
learnr::run_tutorial("04-lec", package = "statmodeling")
learnr::run_tutorial("04-lab", package = "statmodeling")
learnr::run_tutorial("05-lec", package = "statmodeling")
learnr::run_tutorial("05-lab", package = "statmodeling")
```

Those who prefer working on the lab exercises outside of the interactive
`learnr` environment (i.e., in a regular R session) can use the `Rmd`
files supplied in the folder `lab-materials`, which contains both
exercises and solutions.

## About the Instructor

Denis Cohen is a postdoctoral fellow in the Data and Methods Unit at the
[Mannheim Centre for European Social Research
(MZES)](https://www.mzes.uni-mannheim.de/), [University of
Mannheim](https://www.uni-mannheim.de/). He is also lead organizer of
the [MZES Social Science Data
Lab](https://www.mzes.uni-mannheim.de/socialsciencedatalab/page/events/)
and lead editor of the blog [Methods
Bites](https://www.mzes.uni-mannheim.de/socialsciencedatalab/). A
political scientist by training, his substantive work focuses on the
political economy of spatial inequalities, political preferences and
voting behavior, strategic elite behavior, and political competition in
consolidated multiparty democracies. His methodological interests
include advanced statistical modeling, georeferenced data, data
visualization, and causal inference.
