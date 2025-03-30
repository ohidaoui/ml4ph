# ML4PH

A bayesian statistical analysis of the World Health Organization (WHO) life expectancy dataset. The objective is to investigate the relationship between various population health indicators and life expectancy across African countries to inform government policy decisions on resource allocation and public health priorities. 

## Implemented models in `Stan`:

- **Bayesina linear regression**.
- **Hierarchical Bayesian linear regression (random-intercept model)**.
- **Multi-dimentional Gaussian Process with Hilbert Space approximation** (Not yet tested).

## Reproduction

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml4ph.git
cd ml4ph
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install the package and its dependencies:
```bash
pip install -e .
```
4. Run analysis in [life_expectancy_analysis.ipynb](src/ml4ph/life_expectancy_analysis.ipynb).


## Project Structure

```
ml4ph/
├── src/ml4ph/
│   ├── bayesian_predictions.py         # Utilities for Bayesian predictions through MC approximations
│   ├── stan/                           # Stan model definitions
│   └── life_expectancy_analysis.ipynb  # Analysis notebook
```


## Author

- ohidaoui (ou.hidaoui@gmail.com)
