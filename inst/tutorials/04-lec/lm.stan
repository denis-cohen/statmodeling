data {
  int<lower=1> N; // num. observations
  int<lower=1> K; // num. predictors
  matrix[N,K] x; // design matrix
  vector[N] y;    // outcome vector
}

parameters {
  vector[K] beta;      // coef vector
  real<lower=0> sigma; // scale parameter
}

transformed parameters {
  vector[N] mu;  // declare lin. pred.
  mu = x * beta; // assign lin. pred.
}

model {
  // priors
  target += normal_lpdf(beta | 0, 10);  // priors for beta
  target += cauchy_lpdf(sigma |0, 5);   // prior for sigma
  
  // log-likelihood
  target += normal_lpdf(y | mu, sigma); // likelihood
}
