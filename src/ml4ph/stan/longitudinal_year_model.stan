data {
  int<lower=1> N;               // total observations
  int<lower=1> J;               // number of countries
  int<lower=1> K;               // number of predictors
  array[N] int<lower=1,upper=J> country; // country indices
  vector[N] year;               // standardized or centered year variable
  matrix[N, K] X;               // predictor variables
  vector[N] y;                  // life expectancy
}

parameters {
  real alpha;                   // global intercept
  real gamma;                   // year effect
  vector[K] beta;               // regression coefficients
  vector[J] u;                  // random intercepts for countries
  real<lower=0.001> sigma;      // residual SD
  real<lower=0.001> sigma_u;    // SD of random intercepts
}

model {
  // Priors
  alpha ~ normal(0, 10);
  gamma ~ normal(0, 5);
  beta ~ normal(0, 5);
  u ~ normal(0, sigma_u);
  sigma ~ cauchy(0, 2.5);
  sigma_u ~ cauchy(0, 2.5);

  // Likelihood
  y ~ normal(alpha + u[country] + gamma * year + X * beta, sigma);
}

generated quantities {
  vector[N] y_hat;
  for (i in 1:N) {
    y_hat[i] = normal_rng(alpha + u[country[i]] + gamma * year[i] + X[i] * beta, sigma);
  }
}
