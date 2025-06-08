# Optimization Project

This project uses Gurobi and Python to solve an optimization problem related to the automated demand response of a household. Thereby the schedules of flexible appliances are optimized with the goal of minimizing total cost with spotmarket data as a reference.

## Structure

- **src/**: Contains additional python files with outsourced functions.
- **data/**: Contains all the data used in this project. The subfolder *discarded data/* contains old data or data that was further adapted to the final data depicted in the folder above.
- **additional notebooks/**: Contains additional notebooks used for data preparation and analysis

## Results

The project incorporates a total of four cases, each divided into two notesbooks for summer (month of august) and winter (month of december) respectively.

- **base case**: Optimizes the schedule of the three appliances: Dishwasher, washing machine, dryer and additionally the charging of an electriv vehicle.
- **1. case**: Builds upon the the base case by including a heat pump.
- **2. case**: Builds upon the 1. case by including pv.
- **3. case**: Builds upon the 2. case by including vehicle-2-grid & vehicle-2-house capabilities.

## Contributors

- Alessio Häseli
- Severin Wälty
- Simon Bernet