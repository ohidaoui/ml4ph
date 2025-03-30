data {
  int<lower=1> N;             // number of observations
  int<lower=1> K;             // number of predictors
  matrix[N, K] X;             // predictors (standardized)
  vector[N] y;                // target variable (life expectancy)
}

parameters {
  real alpha;                 // intercept
  vector[K] beta;             // regression coefficients
  real<lower=0> sigma;        // error standard deviation
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 5);
  sigma ~ cauchy(0, 2.5);

  // Likelihood
  y ~ normal(alpha + X * beta, sigma);
}