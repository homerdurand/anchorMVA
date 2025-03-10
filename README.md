# anchorMVA

This repository contains code and notebooks for the paper submitted to AISTATS, focusing on anchor regularization and related methodologies for robust statistical learning.

## Repository Structure

### Notebooks
The following Jupyter notebooks are included for experimental analysis and figure generation:

- **`Air_quality_task.ipynb`**: Notebook for the Air Quality prediction task. The dataset can be downloaded from the [Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality).
- **`CMIP56_data_analysis.ipynb`**: Data analysis for the Climate Prediction task, including the generation of Figure 3. The dataset is available upon request from the corresponding authors.
- **`Climate_prediction_task.ipynb`**: Implementation of the Climate Prediction task. Data can be obtained from the corresponding authors.
- **`Pareto_high-dimensionality.ipynb`**: Notebook for generating Figure 6.
- **`perturbation_robustness_IV.ipynb`**: Notebook for generating Figure 2.
- **`perturbation_robustness_IV_high_dimensionality.ipynb`**: Notebook for generating Figure 5.

### Python Code
The core implementation files include:

- **`AnchorOptimalProjector.py`**: Implementation of anchor regularization.
- **`CVP.py`**: Code for Conditional Variance Penalties.
- **`IRM.py`**: Code for Invariant Risk Minimization.
- **`MVA_algo.py`**: Core implementation of the Multi-Variate Algorithm.
- **`data_treatment_tools.py`**: Utility functions for data preprocessing.
- **`reduced_rank_regression.py`**: Implementation of Reduced Rank Regression (RRR).
- **`toy_models.py`**: Code for generating toy models used in experiments.

## Usage
To run the notebooks, ensure you have the required dependencies installed. We recommend using a virtual environment and installing dependencies via:

```bash
pip install -r requirements.txt
```

For data access related to the Climate Prediction task, please contact the corresponding authors.


