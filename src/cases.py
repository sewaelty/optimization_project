import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import sys
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def dishwasher(Time_interval,merged_data,model):
    # Dishwasher properties
    duration = 3  # hours of operation
    min_gap = 15  # hours between runs
    #power_dishwasher = 1.5  # kW during operation

    # Binary start variables: 1 if dishwasher starts at hour t
    start_times = range(Time_interval)# - duration + 1)
    dishwasher_start = model.addVars(Time_interval, vtype=GRB.BINARY, name="start_dw")

    # Binary on variables: 1 if dishwasher is on at hour t
    binary_dishwasher = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_dw")

    # When dishwasher is on, it must be running -> dw_start is 1 -> binary_dishwasher at the same time + duration_dw is 1 
    for t, k in itertools.product(range(len(start_times)-duration + 1), range(duration)):
        model.addConstr(binary_dishwasher[t + k] >= dishwasher_start[t], name=f"dishwasher_{t}_{k}")
    for t in range(len(start_times) - duration):
        model.addConstr(gp.quicksum(binary_dishwasher[t + k] for k in range(duration + 1)) <= 3, name=f"dw_max_three_hours_on_{t}")

    # Enforce min 1 run per day
    # adds up all possible start times of the dishwasher in a single and '>= 1' day ensures that the dishwasher has to run once per day
    hours_per_day = 24
    days = Time_interval // hours_per_day
    for d in range(days):
        model.addConstr(gp.quicksum(dishwasher_start[t] for t in range(d * hours_per_day, (d + 1) * hours_per_day)) == 1,
                    name=f"min_one_run_per_day_dishwasher_{d}")

    # Enforce minimum gap (15 hours) between two starts
    # multiplying with min_gap so we don't constrain the dishwasher to start at the same time every day
    # this enures that the optimizer can iterate through without being constrained to a single time
    for t in range(len(start_times)-min_gap):
        model.addConstr(gp.quicksum(dishwasher_start[t + offset] for offset in range(1, min_gap + 1)) <= (1 - dishwasher_start[t]) * min_gap,
                    name=f"min_gap_after_dw_{t}")
    
    # constraint that the dishwasher may not start in the last 3 hours of the entire time interval
    for t in range(Time_interval - duration, Time_interval):
        model.addConstr(dishwasher_start[t] == 0, name=f"no_start_in_last_hours_{t}")
    model.update()
    return binary_dishwasher, dishwasher_start, start_times
    

def washing_machine(Time_interval,merged_data,model):
    # washing_machine properties
    duration_wm = 2  # hours of operation
    min_gap_wm = 1  # hours between runs
    #power_wm = 3  # kW during operation 
    wm_runs_per_week = 4

    # Binary start variables: 1 if washing machine starts at hour t
    start_times_wm = range(Time_interval - duration_wm + 1)
    wm_start = model.addVars(start_times_wm, vtype=GRB.BINARY, name="start_wm")

    # Binary on variables: 1 if washing machine is on at hour t
    binary_wm = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_wm")

    # When washing machine is on, it must be running -> wm_start is 1 -> binary_wm at the same time + duration_wm is 1 
    for t, k in itertools.product(range(len(start_times_wm)), range(duration_wm)):
        model.addConstr(binary_wm[t + k] >= wm_start[t], name=f"wm_{t}_{k}")
    for t in range(len(start_times_wm)-1):
        model.addConstr(gp.quicksum(binary_wm[t + k] for k in range(duration_wm+1)) <= duration_wm, name=f"wm_max_two_hours_on_{t}")

    # Enforce min 4 runs per week -> if negative prices, can run more than 4 times, for now: exactly 4 times
    # adds up all possible start times of the washing machine in a single week and '>= 1' day ensures that the washing machine has to run 4 times per week
    hours_per_week = 24*7
    weeks = Time_interval // hours_per_week
    for week in range(weeks):
        model.addConstr(gp.quicksum(wm_start[t] for t in range(week * 24 * 7, (week + 1) * 24 * 7 -  1)) == wm_runs_per_week,
                    name=f"wm_four_runs_per_week_{week}")

    # Enforce minimum gap (1 hour) between two starts
    for t in range(len(start_times_wm)-min_gap_wm):
        model.addConstr(gp.quicksum(wm_start[t + offset] for offset in range(1, min_gap_wm + 1)) <= (1 - wm_start[t]) * min_gap_wm,
                    name=f"min_gap_after_washing_wm_{t}")
    
    # Add a new column to the dataframe indicating when the washing machine can be turned on
    merged_data['Allowed_WM_summer'] = 0  # Initialize with 0

    #washing machine can only run during certain hours of the day, during the week after 4pm and on weekends after 10am
    for index, row in merged_data.iterrows():
        hour = row['timestamp'].hour
        day_of_week = row['timestamp'].weekday()  # Monday=0, Sunday=6
        if (day_of_week < 5 and 16 <= hour < 20) or (day_of_week >= 5 and 8 <= hour < 20):
            merged_data.loc[index, 'Allowed_WM_summer'] = 1
        else: 
            merged_data.loc[index, 'Allowed_WM_summer'] = 0
        
    model.addConstrs((binary_wm[t] <= merged_data['Allowed_WM_summer'][t] for t in range(len(binary_wm))), name="allowed_wm_summer")

    model.update()
    return duration_wm, wm_start, binary_wm, start_times_wm
    
def dryer(Time_interval,merged_data,model,duration_wm, wm_start):
    # dryer properties
    duration_dryer = 2  # hours of operation
    max_gap_wm_dryer = 2  # hours between washing machine end and dryer start
    #power_dryer = 3  # kW during operation 

    # Binary start variables: 1 if dryer starts at hour t
    start_times_dryer = range(Time_interval - duration_dryer + 1)
    dryer_start = model.addVars(start_times_dryer, vtype=GRB.BINARY, name="start_dryer")

    # Binary on variables: 1 if dryer is on at hour t
    binary_dryer = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_dryer")

    # When dryer is on, it must be running -> dryer_start is 1 -> binary_dryer at the same time + duration_dryer is 1 
    for t, k in itertools.product(range(len(start_times_dryer)), range(duration_dryer)):
        model.addConstr(binary_dryer[t + k] >= dryer_start[t], name=f"dryer_{t}_{k}")
    for t in range(len(start_times_dryer)-1):
        model.addConstr(gp.quicksum(binary_dryer[t + k] for k in range(duration_dryer+1)) <= duration_dryer, name=f"dryer_max_two_hours_on_{t}")

    # Enforce that the dryer starts within max_gap_wm_dryer hours after the washing machine ends
    for t in range(len(dryer_start)-max_gap_wm_dryer-duration_wm):
        model.addConstr(gp.quicksum(dryer_start[t + offset + duration_wm] for offset in range(0, max_gap_wm_dryer + 1)) >= wm_start[t],
                    name=f"max_gap_after_washing_dryer_{t}")

    #washing machine can only run during certain hours of the day, during the week after 4pm and on weekends after 10am
    # Add a new column to the dataframe indicating when the washing machine can be turned on
    merged_data['Allowed_dryer_summer'] = 0  # Initialize with 0

    # Define the conditions for weekdays and weekends -> for summer and winter
    for index, row in merged_data.iterrows():
        hour = row['timestamp'].hour
        day_of_week = row['timestamp'].weekday()  # Monday=0, Sunday=6
        if (day_of_week < 5 and 16 <= hour < 22) or (day_of_week >= 5 and 8 <= hour < 22):
            merged_data.at[index, 'Allowed_dryer_summer'] = 1
        else: 
            merged_data.at[index, 'Allowed_dryer_summer'] = 0

    model.addConstrs((binary_dryer[t] <= merged_data['Allowed_dryer_summer'][t] for t in range(len(binary_dryer))), name="allowed_dryer_summer")

    model.update()
    return binary_dryer, start_times_dryer, dryer_start
    
def EV_no_feed_in(Time_interval,merged_data,model):
    # EV properties
    min_power_ev = 1 #kW, minimum power to charge the EV
    max_power_ev = 10 #kW, maximum power to charge the EV
    kwh_per_km = 0.2  # kWh per km driven
    max_capacity_ev = 70 #kWh

    ###Variables
    #state of charge of the EV at each time step
    soc_ev = model.addVars(Time_interval,lb=0, ub=max_capacity_ev, vtype=GRB.CONTINUOUS, name="soc_ev")
    #how much power is being charged at each time step
    charging_ev = model.addVars(Time_interval,lb=0, ub=max_power_ev, vtype=GRB.CONTINUOUS, name="charging_lvl_ev")
    #binary variable to indicate if the EV is being charged at each time step
    binary_ev = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_ev")

    # Enforce SoC at 7:00 every day (hour 7 of each day)
    for d in range(Time_interval // 24):
        t = d * 24 + 7  # 7:00 each day
        if t < Time_interval:
            model.addConstr(soc_ev[t] >= 0.8*max_capacity_ev, name=f"ev_soc_7am_day_{d}")

    # car can only charge if it is at home
    model.addConstrs((binary_ev[t] <= merged_data['ev_at_home_binary'][t] for t in range(len(binary_ev))), name="allowed_ev_summer")

    # if car is at home, it can charge, but not more than the maximum power
    # and if it is charging, it must be charging at least the minimum power
    model.addConstrs((charging_ev[t] <= max_power_ev * merged_data['ev_home_availability'][t] * binary_ev[t] for t in range(Time_interval)), name="max_power_ev")
    model.addConstrs((charging_ev[t] >= min_power_ev * merged_data['ev_home_availability'][t] * binary_ev[t] for t in range(Time_interval)), name="min_power_ev")

    # Constrain ev storage
    initial_soc_ev = 20
    model.addConstr(soc_ev[0] == initial_soc_ev, name="ev_soc_initial")
    model.addConstrs((soc_ev[t] == soc_ev[t-1] + charging_ev[t-1] - merged_data['distance_driven'][t-1] * kwh_per_km for t in range(1,Time_interval)),name="ev_soc_update")

    model.update()
    return charging_ev, soc_ev, binary_ev


def peak_prices(Time_interval,merged_data,model,
                inflexible_demand, binary_wm, binary_dishwasher, binary_dryer, 
                charging_ev, power_wm, power_dishwasher, max_power_ev, power_dryer, max_power_hp,power_hp):
    ε = 1e-3
    wanted_steps = 6
    max_demand = max(inflexible_demand) + power_dishwasher + power_wm + power_dryer + max_power_ev + max_power_hp
    levels = np.arange(0, max_demand/7*6 + ε, max_demand/((wanted_steps - 1)))
    levels = np.append(levels, max_demand * 1.5 + 5)  # Ensure coverage

    #multiplier for penalty costs per level increasing by 0.1 per level
    multiplier_per_level = [0.003 * i for i in range(len(levels)-1)]
    M_price = max_demand + 10

    # Binary indicators per level per timestep
    level_bin = [
        [model.addVar(vtype=GRB.BINARY, name=f"level_bin[{t},{i}]") for i in range(len(levels)-1)]
        for t in range(Time_interval)
    ]

    # Integer index of active level
    #demand_level = [
        #model.addVar(lb=0, ub=len(levels) - 1, vtype=GRB.INTEGER, name=f"demand_level[{t}]")
        #for t in range(Time_interval)
    #]

    # Total demand per timestep
    total_demand = model.addVars(Time_interval, lb=0, ub=max_demand, vtype=GRB.CONTINUOUS, name="total_demand")

    # Constraint: calculate total demand
    for t in range(Time_interval):
        model.addConstr(
            total_demand[t] ==
            merged_data['Inflexible_Demand_(kWh)'][t] +
            power_dishwasher * binary_dishwasher[t] +
            power_wm * binary_wm[t] +
            power_dryer * binary_dryer[t] +
            charging_ev[t] + power_hp
        )

    # Constraint: only one level active at a time
    for t in range(Time_interval):
        model.addConstr(gp.quicksum(level_bin[t]) == 1, name=f"one_level_active_{t}")

    # Constraint: bind total_demand to its level using Big-M
    for t in range(Time_interval):
        for i in range(len(levels) - 1):
            model.addConstr(
                total_demand[t] >= levels[i] - (1 - level_bin[t][i]) * M_price,
                name=f"lower_bound_level_{t}_{i}"
            )
            model.addConstr(
                total_demand[t] <= levels[i + 1] - ε + (1 - level_bin[t][i]) * M_price,
                name=f"upper_bound_level_{t}_{i}"
            )
        # Calculate demand_level from binary selection
        #model.addConstr(
            #demand_level[t] == gp.quicksum(i * level_bin[t][i] for i in range(len(levels)-1)),
            #name=f"demand_level_calc_{t}"
        #)

    #generate penalty costs for each level that depend on a const and a linear term
    penalty_per_level = [multiplier_per_level[i] * levels[i] for i in range(len(levels)-1)]

    # Penalty term as an expression
    penalty_cost = gp.quicksum(
        penalty_per_level[i] * level_bin[t][i]
        for t in range(Time_interval)
        for i in range(len(levels)-1)
    )

    model.update()
    return penalty_cost, level_bin, levels, penalty_per_level, total_demand

def heat_pump(Time_interval,merged_data,model, max_power_hp):
    # Heat pump and storage parameters
    # https://www.ochsner.com/de-ch/ochsner-produkte/air-11-c11a/
    COP = 4.2  # Coefficient of Performance
    storage_capacity = 200  # kWh, thermal storage capacity
    storage_loss_rate = 0.01  # 1% loss per hour

    #binary_hp = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_hp")
    power_hp = model.addVars(Time_interval, lb=0, ub=max_power_hp, vtype=GRB.CONTINUOUS, name="power_hp")
    heat_output = model.addVars(Time_interval, lb=0, ub=COP*max_power_hp, vtype=GRB.CONTINUOUS, name="heat_output")
    storage_level = model.addVars(Time_interval, lb=0.2*storage_capacity, ub=storage_capacity, vtype=GRB.CONTINUOUS, name="heat_storage")

    # Heat output from heat pump
    for t in range(Time_interval):
        model.addConstr(heat_output[t] == COP * power_hp[t], name=f"heat_output_{t}")

    # Storage level dynamics
    for t in range(Time_interval):
        if t == 0:
            model.addConstr(
                storage_level[t] == 0.5 * storage_capacity, name=f"storage_balance_{t}"
            )
        else:
            model.addConstr(
                storage_level[t] == storage_level[t - 1] * (1 - storage_loss_rate) + heat_output[t] - merged_data['Heating_Demand_(kWh)'][t],
                name=f"storage_balance_{t}"
            )

    model.update()
    return power_hp

def PV_no_feed_in_and_penalty(Time_interval,merged_data_summer_case_2,model,merged_data,power_dishwasher,binary_dishwasher
                            ,power_wm,binary_wm,power_dryer,binary_dryer,charging_ev,inflexible_demand,max_power_ev,max_power_hp,power_hp):
    # for power produced with the PV system, the price is 0 
    # Total power consumption including fixed and dishwasher
    total_load = {
        t: merged_data['Inflexible_Demand_(kWh)'][t] +
        power_dishwasher * binary_dishwasher[t] +
        power_wm * binary_wm[t] + 
        power_dryer * binary_dryer[t] + 
        charging_ev[t]+ power_hp[t]
        for t in range(Time_interval)
    }

    # Binary variable to indicate if PV production is maxed out
    pv_maxed_binary = model.addVars(Time_interval, vtype=GRB.BINARY, name="pv_maxed")

    # Choose M large enough to cover max difference between pv and load
    M = 10000

    # If demand is higher than PV production
    unmet   = model.addVars(Time_interval, lb=0.0, name="unmet_load")

    # If PV production is higher than demand, the excess is curtailed
    curtail = model.addVars(Time_interval, lb=0.0, name="curtail_pv")

    # Add constraints for PV production, unmet load, and curtailment
    for t in range(Time_interval):
        pv = merged_data['PV_energy_production_kWh'][t]
        load = total_load[t]
        
        # Binary switch: if PV > load → binary = 0; else 1
        model.addConstr(pv - load + unmet[t] - curtail[t] == 0, name=f"pv_load_balance_{t}")
        model.addConstr(curtail[t] <= (1-pv_maxed_binary[t]) * M , name=f"curtail_pv_{t}_2")
        model.addConstr(unmet[t] <= pv_maxed_binary[t] * M, name=f"unmet_load_{t}_2")

    ε = 1e-3
    wanted_steps = 6
    max_demand = max(inflexible_demand) + power_dishwasher + power_wm + power_dryer + max_power_ev
    levels = np.arange(0, max_demand/7*6 + ε, max_demand/((wanted_steps - 1)))
    levels = np.append(levels, max_demand * 1.5 + 5)  # Ensure coverage

    #multiplier for penalty costs per level increasing by 0.1 per level
    multiplier_per_level = [0.003 * i for i in range(len(levels)-1)]
    M_price = max_demand + 10

    # Binary indicators per level per timestep
    level_bin = [
        [model.addVar(vtype=GRB.BINARY, name=f"level_bin[{t},{i}]") for i in range(len(levels)-1)]
        for t in range(Time_interval)
    ]

    # Integer index of active level
    demand_level = [
        model.addVar(lb=0, ub=len(levels) - 1, vtype=GRB.INTEGER, name=f"demand_level[{t}]")
        for t in range(Time_interval)
    ]

    # Constraint: only one level active at a time
    for t in range(Time_interval):
        model.addConstr(gp.quicksum(level_bin[t]) == 1, name=f"one_level_active_{t}")

    # Constraint: bind net_demand to its level using Big-M
    for t in range(Time_interval):
        for i in range(len(levels) - 1):
            model.addConstr(
                unmet[t] >= levels[i] - (1 - level_bin[t][i]) * M_price,
                name=f"lower_bound_level_{t}_{i}"
            )
            model.addConstr(
                unmet[t] <= levels[i + 1] - ε + (1 - level_bin[t][i]) * M_price,
                name=f"upper_bound_level_{t}_{i}"
            )
        # Calculate demand_level from binary selection
        model.addConstr(
            demand_level[t] == gp.quicksum(i * level_bin[t][i] for i in range(len(levels)-1)),
            name=f"demand_level_calc_{t}"
        )

    #generate penalty costs for each level that depend on a const and a linear term
    penalty_per_level = [multiplier_per_level[i] * levels[i] for i in range(len(levels)-1)]

    # Penalty term as an expression
    penalty_cost = gp.quicksum(
        penalty_per_level[i] * level_bin[t][i]
        for t in range(Time_interval)
        for i in range(len(levels)-1)
    )


    model.update( )

    
    return pv, load, unmet, curtail, pv_maxed_binary, total_load,penalty_cost, level_bin, levels, penalty_per_level, demand_level

