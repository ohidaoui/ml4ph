functions {
  // Squared-exponential spectral density function.
  vector spd_se(vector omega, real sigma, real ell) {
    return sigma^2 * sqrt(2 * pi()) * ell * exp(-0.5 * (ell * omega) .* (ell * omega));
  }
  
  // Compute the eigenvalues for a 1D domain with boundary L.
  vector eigenvalues(int M, real L) {
    vector[M] lambda;
    for (m in 1:M) {
      lambda[m] = square(m * pi() / (2 * L));
    }
    return lambda;
  }
  
  // Compute the eigenvectors (basis functions) for a 1D input vector x.
  matrix eigenvectors(vector x, int M, real L, vector lambda) {
    int N = num_elements(x);
    matrix[N, M] PHI;
    for (m in 1:M) {
      PHI[, m] = sqrt(1 / L) * sin(sqrt(lambda[m]) * (x + L));
    }
    return PHI;
  }
  
  // HSGP approximation for a 1D input using the SE kernel.
  vector hsgp_se(vector x, real sigma, real ell, vector lambdas, matrix PHI, vector z) {
    vector[num_elements(lambdas)] spds = spd_se(sqrt(lambdas), sigma, ell);
    matrix[num_elements(lambdas), num_elements(lambdas)] Delta = diag_matrix(sqrt(spds));
    return PHI * Delta * z;
  }
}

data {
  int<lower=1> N;                      // Total observations.
  int<lower=1> J;                      // Number of countries.
  int<lower=1> K;                      // Number of predictors.
  array[N] int<lower=1,upper=J> country; // Country indices.
  vector[N] year;                      // Standardized/centered year.
  matrix[N, K] X;                      // Standardized predictor variables.
  vector[N] y;                         // Life expectancy.
  
  real<lower=0> C;                     // Constant for GP domain boundary scaling.
  int<lower=1> M;                      // Number of basis functions per GP dimension.
}

transformed data {
  // Construct the GP input matrix: first column for year, next K columns for X.
  int D = K + 1; // Total GP dimensions: year + predictors.
  matrix[N, D] X_gp;
  for (n in 1:N) {
    X_gp[n, 1] = year[n];
    for (k in 1:K)
      X_gp[n, k+1] = X[n, k];
  }
  
  // For each GP input dimension, compute the domain boundary, eigenvalues, and eigenvectors.
  vector[D] L;
  array[D] vector[M] lambdas;  // New syntax for array of vectors.
  array[D] matrix[N, M] PHI;   // New syntax for array of matrices.
  for (d in 1:D) {
    L[d] = C * (max(abs(X_gp[, d])) + 1e-6);  // Use abs instead of fabs.
    lambdas[d] = eigenvalues(M, L[d]);
    PHI[d] = eigenvectors(X_gp[, d], M, L[d], lambdas[d]);
  }
}

parameters {
  real alpha;                          // Global intercept.
  vector[J] u;                         // Country-specific random intercepts.
  real<lower=0.001> sigma;             // Residual standard deviation.
  real<lower=0.001> sigma_u;           // SD for country random effects.
  
  // GP hyperparameters for each of the D = (K+1) dimensions.
  vector<lower=0>[K+1] sigma_gp;
  vector<lower=0>[K+1] ell_gp;
  
  // Latent coefficients for the HSGP representation (using new array syntax).
  array[K+1] vector[M] z;
}

transformed parameters {
  // The additive GP function is the sum of contributions from each GP dimension.
  vector[N] f_gp = rep_vector(0.0, N);
  for (d in 1:(K+1)) {
    f_gp += hsgp_se(to_vector(X_gp[, d]), sigma_gp[d], ell_gp[d], lambdas[d], PHI[d], z[d]);
  }
  
  // Overall mean: flexible intercept + country random effects + additive GP.
  vector[N] mu = alpha + u[country] + f_gp;
}

model {
  // Priors
  alpha ~ normal(mean(y), 10);
  u ~ normal(0, sigma_u);
  sigma ~ normal(0, 5);
  sigma_u ~ normal(0, 5);
  
  // More informative priors for GP parameters
  // sigma_gp ~ normal(0, 2);
  // ell_gp ~ normal(1, 0.5);
  
  // Priors for GP hyperparameters.
  sigma_gp ~ inv_gamma(5, 5);
  ell_gp ~ inv_gamma(5, 5);
  for (d in 1:(K+1))
    z[d] ~ normal(0, 1);
    
  // Likelihood.
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_hat;
  for (n in 1:N)
    y_hat[n] = normal_rng(mu[n], sigma);
}
