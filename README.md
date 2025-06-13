# Optimization Project

This project uses **Gurobi** and **Python** to solve an optimization problem related to **automated demand response** in a household. The schedules of flexible appliances are optimized with the goal of **minimizing total electricity costs**, using **spot market data** as a reference.

---

## 📁 Project Structure

- **`data/`**: Contains all datasets used in the project. The subfolder `_discarded data/` contains outdated or intermediate versions of datasets.
- **`EV-data/`**: Contains a notebook that analyzes simulation results from _emobpy_ and selects an electric vehicle profile.
- **`main/`**: Contains the main notebook which runs the optimization for all cases described in the _Results_ section. The individual cases are called through functions defined in the **case_functions.py** file in the folder `src`.
- **`results/`**: Contains the csv files with the results.
- **`figures/`**:  Contains the figures used in the report.
- **`src/`**: Includes Python file **case_functions.py** which defines the functions used in the optimization. Also includes files with helper functions and modularized logic. .
- **`additional notebooks/`**: Contains supplementary notebooks used for data preparation and exploratory analysis.
- **`requirements.txt`**: Lists all Python packages required to run the code.

---

## 🚀 Setup Instructions

To get started, set up a virtual environment (optional but recommended) and install dependencies:

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

---

## 📊 Results

The project is divided into **four cases**, each evaluated for both **summer (August)** and **winter (December)** scenarios:

1. **Base Case**: Optimizes the operation of three flexible appliances (dishwasher, washing machine, dryer) and the charging schedule of an electric vehicle.
2. **Case 1**: Adds a **heat pump** to the base case. Evaluated only in winter due to the assumption of no heating demand in summer.
3. **Case 2**: Builds on Case 1 by integrating **photovoltaic (PV) generation**. Surplus PV energy is curtailed if not used.
4. **Case 3**: Extends Case 2 by including **vehicle-to-grid (V2G)** and **vehicle-to-home (V2H)** capabilities.

---

## 👥 Contributors

- Alessio Häseli
- Severin Wälty
- Simon Bernet
```
