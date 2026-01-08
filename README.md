# Stability of Fault Predictions Using Triangulated Models of Homoclinal Interfaces

## Context & Background
This repository extends the research presented in **[SubsurfaceBreaks v. 1.0 (Michał Michalak, 2025)](https://github.com/michalmichalak997/SubsurfaceBreaks)**. It introduces and evaluates two additional data scaling methods to assess the stability of fault predictions and to explore alternative approaches that yield specific results.

## Methodology Overview
This project compares three distinct approaches to data scaling (referred to as `SS_MODE` in the code):

* **Method 1 (Original):** The global scaling technique used in the original research.
* **Method 2 (Per-Surface):** A localized scaling technique applied individually to each geological surface/file. This method requires an ordered split of data to prevent leakage.
* **Method 3 (Standard ML):** A standard machine learning approach where the scaler is fitted strictly on the training set and applied to the test set.

## Repository Content (additional)

### Experiments
* **`DataScalingMethods/ss_experiment_1_check_if_stable.ipynb`**
    * **Active Modes:** Method 1 & Method 3.
    * Performs the 'stability' experiment using random train-test splits.
* **`DataScalingMethods/ss_experiment_4_check_if_stable_ordered_split_downsampling.ipynb`**
    * **Active Mode:** Method 2.
    * Performs the 'stability' experiment using **ordered splits** to accommodate the per-surface scaling logic.

### Statistical Analysis
* **`DataScalingMethods/ss_experiment_0_scrap_data.ipynb`**
    * Loads input data and generates raw datasets (variables) required for the statistical analysis.
* **`DataScalingMethods/ss_experiment_2_compare_parameters.ipynb`**
    * Performs statistical analysis of the transformed data (normality tests, variance comparison, mean equality).
    * **Note:** Preliminary results indicate that normality assumptions are violated, affecting the interpretation of standard parametric tests (e.g., Student's t-test).

## Experiment Workflow

### Core Logic
The workflow follows the pipeline established in the parent repository:
1.  **Preprocessing:** Data cleaning, feature extraction (neighbor sorting), and selection.
2.  **Scaling (`SS_MODE`):** The key variable in this research.
    * *Notebook 1* handles global approaches (Modes 1 & 3).
    * *Notebook 4* handles the local approach (Mode 2), necessitating a different splitting strategy (Ordered Split) to avoid data leakage.
3.  **Modeling:** Support Vector Machine (SVM) is used as the baseline classifier.
4.  **Stability Testing:** The model predicts faults on a real-world dataset (Homocline vs. Fault). A subset of this data is then isolated, scaled, and predicted again. Discrepancies between the global prediction and the subset prediction are flagged as "unstable."

### Data Scaling Schemas
The following diagrams illustrate the data flow for each scaling method derived from the original and alternative proposals.

**Method 1 (Global / Original)**
<img width="1045" alt="Method 1 Schema" src="https://github.com/user-attachments/assets/b27c562e-b816-43fc-a735-f5b3a2dbc83f" />

**Method 2 (Per-Surface / Local)**
<img width="1045" alt="Method 2 Schema" src="https://github.com/user-attachments/assets/b405a5be-7cba-4469-b922-162fd2ca0a5c" />

**Method 3 (Standard ML Pipeline)**
<img width="1045" alt="Method 3 Schema" src="https://github.com/user-attachments/assets/e49664d6-bdf3-45e2-840f-17523299e9ab" />

## Visualization of Results
The experiment produces visualizations identifying "unstable" observations-points where the model's classification changes depending on the context (entire dataset vs. subset).

<img width="1229" alt="Stability Result Example" src="https://github.com/user-attachments/assets/75bac653-b251-48bf-8f57-61a45c50e73b" />

## Configuration Constants
The experiments are controlled by the following constants, ensuring reproducibility and consistency with the original research:
* `INPUT_PATH`, `REAL_DATA_FILE`: References to the original datasets.
* `RANDOM_STATE`: Set to `42`.
* `SS_MODE`: Selects the standard scaling method (1, 2, or 3).
* Bounding Box Constants (`X_C_LOWER`, etc.): Define the spatial subset used for stability testing.

## Critical Implementation Notes
To ensure exact reproducibility and maintain logic separation, the following technical constraints apply:

1.  **Data Loading Order:** Input files are loaded using a strict numerical iterator (`range(1000)`) rather than default filesystem alphabetical ordering. This ensures the data sequence matches the original paper exactly, which is crucial for the split logic.
2.  **Model Selection (Baseline vs. Grid Search):** The code includes a complete implementation of `GridSearchCV`. However, it is **not executed**. To ensure the results are comparable with the parent research, the model hyperparameters are hardcoded (they are identified as optimal in the original paper).
3.  **Downsampling Strategy:**
    * *Notebook 1 (Global):* Uses standard random downsampling applied to the entire dataset.
    * *Notebook 4 (Local):* Uses a per-file downsampling strategy to maintain class balance within specific geological regions, which is necessary for the Per-Surface scaling method.
4.  **Controlled Code Redundancy:** The notebooks contain implementation logic for valid scaling modes from other notebooks (e.g., Method 2 code exists in Notebook 1). This is a known technical debt retained for reference. It is kept **under strict control** via assertion cells (e.g., `raise ValueError`) at the beginning of execution, preventing the use of incompatible methods in the wrong environment.
