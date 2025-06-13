import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools


def dishwasher(Time_interval, model):
    """
    In the following function the dishwasher is implemented.
    This function is called in all cases
    """

    # Dishwasher properties
    duration = 3  # Duration the dishwasher runs once started in h
    min_gap = 15  # Minimum number of hours required between two dishwasher start events

    # Time indices representing all discrete time steps in the optimization horizon
    start_times = range(Time_interval)
    # Variable: Binary decision variables: 1 if the dishwasher starts at time t, 0 otherwise
    dishwasher_start = model.addVars(Time_interval, vtype=GRB.BINARY, name="start_dw")
    # Variable: Binary state variables: 1 if the dishwasher is running at time t, 0 otherwise
    binary_dishwasher = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_dw")

    # Constraint: Link start and run variables: if the dishwasher starts at t, it must be running for the next 'duration' hours
    for t, k in itertools.product(
        range(len(start_times) - duration + 1), range(duration)
    ):
        model.addConstr(
            binary_dishwasher[t + k] >= dishwasher_start[t], name=f"dishwasher_{t}_{k}"
        )

    # Constraint: Limit the dishwasher to a maximum of 3 consecutive running hours (enforces correct shutdown behavior; important when negative prices are possible)
    for t in range(len(start_times) - duration):
        model.addConstr(
            gp.quicksum(binary_dishwasher[t + k] for k in range(duration + 1)) <= 3,
            name=f"dw_max_three_hours_on_{t}",
        )

    # Constraint: Enforce exactly one dishwasher start per day
    hours_per_day = 24
    days = Time_interval // hours_per_day
    for d in range(days):
        model.addConstr(
            gp.quicksum(
                dishwasher_start[t]
                for t in range(d * hours_per_day, (d + 1) * hours_per_day)
            )
            == 1,
            name=f"min_one_run_per_day_dishwasher_{d}",
        )

    # Constraint: Ensure a minimum gap of 15 hours between any two dishwasher starts to avoid overlapping cycles
    for t in range(len(start_times) - min_gap):
        model.addConstr(
            gp.quicksum(
                dishwasher_start[t + offset] for offset in range(1, min_gap + 1)
            )
            <= (1 - dishwasher_start[t]) * min_gap,
            name=f"min_gap_after_dw_{t}",
        )

    # Constraint: Prohibit dishwasher starts in the last 'duration' hours, as the full cycle can't be completed (Embedded in T_start,dw in mathematical formulation)
    for t in range(Time_interval - duration, Time_interval):
        model.addConstr(dishwasher_start[t] == 0, name=f"no_start_in_last_hours_{t}")

    model.update()
    # return variables needed for further steps and/or plots
    return binary_dishwasher, dishwasher_start, start_times


def washing_machine(Time_interval, merged_data, model):
    """
    In the following function the washing machine is implemented.
    This function is called in all cases
    """

    # Washing machine properties
    duration_wm = 2  # Duration of one washing cycle in hours
    min_gap_wm = 1  # Minimum number of hours between two washing machine runs
    wm_runs_per_week = 4  # Required number of washing machine runs per week

    # Define time indices where washing machine can start (excluding the last hours where full cycle can't fit)
    start_times_wm = range(Time_interval - duration_wm + 1)

    # Variable: Binary decision variables: 1 if the washing machine starts at hour t, 0 otherwise
    wm_start = model.addVars(start_times_wm, vtype=GRB.BINARY, name="start_wm")

    # Variable: Binary state variables: 1 if the washing machine is on at hour t, 0 otherwise
    binary_wm = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_wm")

    # Constraint: Link start and running states: if started at t, the machine must be on for the next 'duration_wm' hours
    for t, k in itertools.product(range(len(start_times_wm)), range(duration_wm)):
        model.addConstr(binary_wm[t + k] >= wm_start[t], name=f"wm_{t}_{k}")

    # Constraint: Limit operation to at most 'duration_wm' consecutive hours to prevent extended unintended runs (prevents making use of negative prices)
    for t in range(len(start_times_wm) - 1):
        model.addConstr(
            gp.quicksum(binary_wm[t + k] for k in range(duration_wm + 1))
            <= duration_wm,
            name=f"wm_max_two_hours_on_{t}",
        )

    # Constraint: Enforce exactly 4 washing machine runs per week
    hours_per_week = 24 * 7
    weeks = Time_interval // hours_per_week
    for week in range(weeks):
        model.addConstr(
            gp.quicksum(
                wm_start[t] for t in range(week * 24 * 7, (week + 1) * 24 * 7 - 1)
            )
            == wm_runs_per_week,
            name=f"wm_four_runs_per_week_{week}",
        )

    # Constraint: Enforce minimum time gap between two consecutive starts (to avoid overlapping runs)
    for t in range(len(start_times_wm) - min_gap_wm):
        model.addConstr(
            gp.quicksum(wm_start[t + offset] for offset in range(1, min_gap_wm + 1))
            <= (1 - wm_start[t]) * min_gap_wm,
            name=f"min_gap_after_washing_wm_{t}",
        )

    # Add a column to the input data indicating allowed hours for washing machine operation
    merged_data["Allowed_WM_summer"] = 0  # Initialize with 0s

    # Define operational hours: weekdays after 4 PM, weekends after 10 AM (until 8 PM both cases)
    for index, row in merged_data.iterrows():
        hour = row["timestamp"].hour
        day_of_week = row["timestamp"].weekday()  # Monday=0, Sunday=6
        if (day_of_week < 5 and 16 <= hour < 20) or (
            day_of_week >= 5 and 8 <= hour < 20
        ):
            merged_data.loc[index, "Allowed_WM_summer"] = 1
        else:
            merged_data.loc[index, "Allowed_WM_summer"] = 0

    # Constraint: Restrict washing machine operation to only allowed hours
    model.addConstrs(
        (
            binary_wm[t] <= merged_data["Allowed_WM_summer"][t]
            for t in range(len(binary_wm))
        ),
        name="allowed_wm_summer",
    )

    model.update()
    # return variables needed for further steps and/or plots
    return duration_wm, wm_start, binary_wm, start_times_wm


def dryer(Time_interval, merged_data, model, duration_wm, wm_start):
    """
    In the following function the dryer is implemented.
    This function is called in all cases
    """

    # Dryer properties
    duration_dryer = 2  # Duration of one drying cycle in hours
    max_gap_wm_dryer = (
        2  # Maximum number of hours allowed between washing machine end and dryer start
    )

    # Define time indices where dryer can start (excluding final hours where full cycle cannot fit)
    start_times_dryer = range(Time_interval - duration_dryer + 1)

    # Variable: Binary decision variables: 1 if dryer starts at hour t, 0 otherwise
    dryer_start = model.addVars(start_times_dryer, vtype=GRB.BINARY, name="start_dryer")

    # Variable: Binary state variables: 1 if dryer is on at hour t, 0 otherwise
    binary_dryer = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_dryer")

    # Constraint: Link start and running states: if started at t, the dryer must be on for the next 'duration_dryer' hours
    for t, k in itertools.product(range(len(start_times_dryer)), range(duration_dryer)):
        model.addConstr(binary_dryer[t + k] >= dryer_start[t], name=f"dryer_{t}_{k}")

    # Constraint: Limit operation to at most 'duration_dryer' consecutive hours to prevent unintended extended runs
    for t in range(len(start_times_dryer) - 1):
        model.addConstr(
            gp.quicksum(binary_dryer[t + k] for k in range(duration_dryer + 1))
            <= duration_dryer,
            name=f"dryer_max_two_hours_on_{t}",
        )

    # Constraint: Ensure that the dryer starts within 'max_gap_wm_dryer' hours after the washing machine finishes
    for t in range(len(dryer_start) - max_gap_wm_dryer - duration_wm):
        model.addConstr(
            gp.quicksum(
                dryer_start[t + offset + duration_wm]
                for offset in range(0, max_gap_wm_dryer + 1)
            )
            >= wm_start[t],
            name=f"max_gap_after_washing_dryer_{t}",
        )

    # Add a column to indicate allowed hours for dryer operation (summer rules)
    merged_data["Allowed_dryer"] = 0  # Initialize with 0s

    # Define operational hours: weekdays from 4 PM to 10 PM, weekends from 8 AM to 10 PM
    for index, row in merged_data.iterrows():
        hour = row["timestamp"].hour
        day_of_week = row["timestamp"].weekday()  # Monday=0, Sunday=6
        if (day_of_week < 5 and 16 <= hour < 22) or (
            day_of_week >= 5 and 8 <= hour < 22
        ):
            merged_data.at[index, "Allowed_dryer"] = 1
        else:
            merged_data.at[index, "Allowed_dryer"] = 0

    # Constraint: Constrain dryer usage to only allowed hours
    model.addConstrs(
        (
            binary_dryer[t] <= merged_data["Allowed_dryer"][t]
            for t in range(len(binary_dryer))
        ),
        name="allowed_dryer",
    )

    model.update()
    # Return key variables for downstream use and/or analysis
    return binary_dryer, start_times_dryer, dryer_start


def EV_no_feed_in(Time_interval, merged_data, model):
    """
    In the following function the EV-charging is implemented.
    This function is called in all cases, except for Case 3!
    """

    # EV properties
    min_power_ev = 1  # Minimum charging power in kW
    max_power_ev = 10  # Maximum charging power in kW
    kwh_per_km = 0.2  # Energy consumption in kWh per km
    max_capacity_ev = 70  # Maximum EV battery capacity in kWh

    # Variable: State of charge (SoC) of the EV battery at each time step
    soc_ev = model.addVars(
        Time_interval, lb=0, ub=max_capacity_ev, vtype=GRB.CONTINUOUS, name="soc_ev"
    )

    # Variable: Charging power applied to the EV at each time step
    charging_ev = model.addVars(
        Time_interval,
        lb=0,
        ub=max_power_ev,
        vtype=GRB.CONTINUOUS,
        name="charging_lvl_ev",
    )

    # Variable: Binary variable: 1 if EV is charging at time t, 0 otherwise
    binary_ev = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_ev")

    # Constraint: Ensure minimum SoC of 80% at 7:00 AM each day
    for d in range(Time_interval // 24):
        t = d * 24 + 7  # Hour 7 (7:00 AM) of each day
        if t < Time_interval:
            model.addConstr(
                soc_ev[t] >= 0.8 * max_capacity_ev, name=f"ev_soc_7am_day_{d}"
            )

    # Constraint: Charging is only allowed if the EV is at home
    model.addConstrs(
        (
            binary_ev[t] <= merged_data["ev_at_home_binary"][t]
            for t in range(len(binary_ev))
        ),
        name="allowed_ev_summer",
    )

    # Constraint: Charging power respects home availability and binary state
    model.addConstrs(
        (
            charging_ev[t]
            <= max_power_ev * merged_data["ev_home_availability"][t] * binary_ev[t]
            for t in range(Time_interval)
        ),
        name="max_power_ev",
    )

    # Constraint: If charging, enforce a minimum charging power
    model.addConstrs(
        (
            charging_ev[t]
            >= min_power_ev * merged_data["ev_home_availability"][t] * binary_ev[t]
            for t in range(Time_interval)
        ),
        name="min_power_ev",
    )

    # Constraint: Initial SoC of the EV
    initial_soc_ev = 20
    model.addConstr(soc_ev[0] == initial_soc_ev, name="ev_soc_initial")

    # Constraint: SoC update equation accounting for charging and driving energy consumption
    model.addConstrs(
        (
            soc_ev[t]
            == soc_ev[t - 1]
            + charging_ev[t - 1]
            - merged_data["distance_driven"][t - 1] * kwh_per_km
            for t in range(1, Time_interval)
        ),
        name="ev_soc_update",
    )

    model.update()

    # Return key decision variables for later use and/or analysis
    return charging_ev, soc_ev, binary_ev


def peak_prices(
    Time_interval,
    merged_data,
    model,
    inflexible_demand,
    binary_wm,
    binary_dishwasher,
    binary_dryer,
    charging_ev,
    power_wm,
    power_dishwasher,
    max_power_ev,
    power_dryer,
    max_power_hp,
    power_hp,
):
    """
    In the following function the peak price penalty is implemented.
    This function is called in all cases, except for Case 2 & 3!
    """

    ε = 1e-3  # Small constant to avoid strict inequality issues
    wanted_steps = 6  # Number of demand levels before final catch-all level

    # Estimate maximum possible demand based on device power ratings
    base_max_demand = (
        max(inflexible_demand)
        + power_dishwasher
        + power_wm
        + power_dryer
        + max_power_ev
    )

    max_demand = (
        max(inflexible_demand)
        + power_dishwasher
        + power_wm
        + power_dryer
        + max_power_ev
        + max_power_hp
    )

    # Create demand levels (bins) and ensure full demand range is covered
    levels = np.arange(
        0, base_max_demand / 7 * 6 + ε, base_max_demand / ((wanted_steps - 1))
    )
    levels = np.append(levels, max_demand + ε)  # Ensure coverage

    # Define a price multiplier for each level (linearly increasing)
    multiplier_per_level = [0.003 * i for i in range(len(levels) - 1)]

    # Big-M value for enforcing level constraints
    M_price = max_demand + 10

    # Variable: Binary variable matrix indicating which demand level is active at each timestep
    level_bin = [
        [
            model.addVar(vtype=GRB.BINARY, name=f"level_bin[{t},{i}]")
            for i in range(len(levels) - 1)
        ]
        for t in range(Time_interval)
    ]

    # Variable: Continuous variable: total power demand at each time step
    total_demand = model.addVars(
        Time_interval, lb=0, ub=max_demand, vtype=GRB.CONTINUOUS, name="total_demand"
    )

    # Constraint: Compute total demand from all sources at time t
    for t in range(Time_interval):
        model.addConstr(
            total_demand[t]
            == merged_data["Inflexible_Demand_(kWh)"][t]
            + power_dishwasher * binary_dishwasher[t]
            + power_wm * binary_wm[t]
            + power_dryer * binary_dryer[t]
            + charging_ev[t]
            + power_hp[t]
        )

    # Constraint: Only one demand level can be active at each time step
    for t in range(Time_interval):
        model.addConstr(gp.quicksum(level_bin[t]) == 1, name=f"one_level_active_{t}")

    # Constraint: Bind total demand to its corresponding level using Big-M method
    for t in range(Time_interval):
        for i in range(len(levels) - 1):
            model.addConstr(
                total_demand[t] >= levels[i] - (1 - level_bin[t][i]) * M_price,
                name=f"lower_bound_level_{t}_{i}",
            )
            model.addConstr(
                total_demand[t] <= levels[i + 1] - ε + (1 - level_bin[t][i]) * M_price,
                name=f"upper_bound_level_{t}_{i}",
            )

    # Compute fixed penalty value for each level (level midpoint * multiplier)
    penalty_per_level = [
        multiplier_per_level[i] * levels[i] for i in range(len(levels) - 1)
    ]

    # Objective expression: sum of penalties incurred across all time steps
    penalty_cost = gp.quicksum(
        penalty_per_level[i] * level_bin[t][i]
        for t in range(Time_interval)
        for i in range(len(levels) - 1)
    )

    model.update()

    # Return useful variables for analysis and/or inclusion in objective
    return penalty_cost, level_bin, levels, penalty_per_level, total_demand


def heat_pump(Time_interval, merged_data, model, max_power_hp):
    """
    In the following function the heat pump is implemted.
    This function corresponds to Case 1 and is called for winter in Cases 1, 2 & 3!
    """

    # Parameters for the heat pump and thermal storage
    COP = 4.2  # Coefficient of Performance: how much thermal energy per kWh electricity
    storage_capacity = 200  # kWh: total capacity of the thermal storage
    storage_loss_rate = 0.01  # 1% hourly loss of stored heat

    # Variable: Power input to the heat pump (in kW), bounded by the max rated power
    power_hp = model.addVars(
        Time_interval, lb=0, ub=max_power_hp, vtype=GRB.CONTINUOUS, name="power_hp"
    )

    # Variable: Resulting heat output from the heat pump at each timestep
    heat_output = model.addVars(
        Time_interval,
        lb=0,
        ub=COP * max_power_hp,
        vtype=GRB.CONTINUOUS,
        name="heat_output",
    )

    # Variable: Thermal storage level at each timestep, bounded by 20%-100% of capacity
    storage_level = model.addVars(
        Time_interval,
        lb=0.2 * storage_capacity,
        ub=storage_capacity,
        vtype=GRB.CONTINUOUS,
        name="heat_storage",
    )

    # Constraint: Link power input to heat output using COP
    for t in range(Time_interval):
        model.addConstr(heat_output[t] == COP * power_hp[t], name=f"heat_output_{t}")

    # Constraint: Storage balance with losses and heat consumption
    for t in range(Time_interval):
        if t == 0:
            model.addConstr(
                storage_level[t] == 0.5 * storage_capacity, name=f"storage_balance_{t}"
            )
        else:
            model.addConstr(
                storage_level[t]
                == storage_level[t - 1] * (1 - storage_loss_rate)
                + heat_output[t]
                - merged_data["Heating_Demand_(kWh)"][t],
                name=f"storage_balance_{t}",
            )

    model.update()

    # Return the power variable for further analyis
    return power_hp


def PV_no_feed_in_and_penalty(
    Time_interval,
    merged_data,
    model,
    power_dishwasher,
    binary_dishwasher,
    power_wm,
    binary_wm,
    power_dryer,
    binary_dryer,
    charging_ev,
    inflexible_demand,
    max_power_ev,
    power_hp,
):
    """
    Implements PV generation without feed-in to the grid and penalizes high peak demand.
    Used in Case 2.
    """
    ### ---- PV Generation ---- ###

    # Calculate total electricity demand at each timestep (includes all flexible and inflexible components)
    total_load = {
        t: merged_data["Inflexible_Demand_(kWh)"][t]
        + power_dishwasher * binary_dishwasher[t]
        + power_wm * binary_wm[t]
        + power_dryer * binary_dryer[t]
        + charging_ev[t]
        + power_hp[t]
        for t in range(Time_interval)
    }

    # Estimate the maximum possible electric demand across all components except Heatpump
    base_max_demand = (
        max(inflexible_demand)
        + power_dishwasher
        + power_wm
        + power_dryer
        + max_power_ev
    )

    # Estimate the maximum possible electric demand across all components
    max_demand = (
        max(inflexible_demand)
        + power_dishwasher
        + power_wm
        + power_dryer
        + max_power_ev
    )

    # Variable: Binary variable: 1 if PV is fully used (demand >= PV), 0 if there is surplus PV
    pv_maxed_binary = model.addVars(Time_interval, vtype=GRB.BINARY, name="pv_maxed")

    # Big-M constant for disjunctive constraints
    M = max(merged_data["PV_energy_production_kWh"]) + max_demand

    # Variables for imbalance: unmet demand > 0 when load > PV and curtailed PV > 0 when PV > load
    unmet = model.addVars(Time_interval, lb=0.0, name="unmet_load")
    curtail = model.addVars(Time_interval, lb=0.0, name="curtail_pv")

    # Balance constraints for every timestep
    for t in range(Time_interval):
        pv = merged_data["PV_energy_production_kWh"][t]
        load = total_load[t]

        # Constraint: Ensure unmet and curtail fill the gap between load and PV
        model.addConstr(
            pv - load + unmet[t] - curtail[t] == 0, name=f"pv_load_balance_{t}"
        )

        # Constraint: Curtailment only allowed if PV > load (pv_maxed_binary = 0)
        model.addConstr(
            curtail[t] <= (1 - pv_maxed_binary[t]) * M, name=f"curtail_pv_{t}_2"
        )

        # Constraint: Unmet only allowed if load > PV (pv_maxed_binary = 1)
        model.addConstr(unmet[t] <= pv_maxed_binary[t] * M, name=f"unmet_load_{t}_2")

    ### ---- PENALTY FOR EXCESSIVE CONSUMPTION ---- ###

    ε = 1e-3  # Define a small buffer to ensure strict inequality handling in level boundaries
    wanted_steps = 6  # Number of discrete penalty tiers you want to model

    # Generate level thresholds (step boundaries)
    levels = np.arange(
        0, base_max_demand / 7 * 6 + ε, base_max_demand / ((wanted_steps - 1))
    )

    # Add a catch-all top level to allow for demand spikes beyond normal range
    levels = np.append(levels, max_demand + ε)

    # Define penalty multiplier per level (linear increase)
    multiplier_per_level = [0.003 * i for i in range(len(levels) - 1)]

    # Big-M constant for disjunctive constraints related to penalty tiers
    M_price = max_demand + 10

    # Variable: Binary indicators for active penalty tier at each timestep
    level_bin = [
        [
            model.addVar(vtype=GRB.BINARY, name=f"level_bin[{t},{i}]")
            for i in range(len(levels) - 1)
        ]
        for t in range(Time_interval)
    ]

    # Variable: Integer index variable to track active tier for logging/visualization
    demand_level = [
        model.addVar(
            lb=0, ub=len(levels) - 1, vtype=GRB.INTEGER, name=f"demand_level[{t}]"
        )
        for t in range(Time_interval)
    ]

    # Constraint: Only one level can be active per timestep
    for t in range(Time_interval):
        model.addConstr(gp.quicksum(level_bin[t]) == 1, name=f"one_level_active_{t}")

    # Constraint: Bind unmet load to its penalty tier using level bounds and Big-M
    for t in range(Time_interval):
        for i in range(len(levels) - 1):
            # Lower bound of level i
            model.addConstr(
                unmet[t] >= levels[i] - (1 - level_bin[t][i]) * M_price,
                name=f"lower_bound_level_{t}_{i}",
            )
            # Upper bound of level i (subtract a buffer to avoid overlap)
            model.addConstr(
                unmet[t] <= levels[i + 1] - ε + (1 - level_bin[t][i]) * M_price,
                name=f"upper_bound_level_{t}_{i}",
            )
        # Map active level_bin combination to integer demand level index
        model.addConstr(
            demand_level[t]
            == gp.quicksum(i * level_bin[t][i] for i in range(len(levels) - 1)),
            name=f"demand_level_calc_{t}",
        )

    # Calculate penalty cost per level based on base level size and multiplier
    penalty_per_level = [
        multiplier_per_level[i] * levels[i] for i in range(len(levels) - 1)
    ]

    # Compute total penalty cost over time
    penalty_cost = gp.quicksum(
        penalty_per_level[i] * level_bin[t][i]
        for t in range(Time_interval)
        for i in range(len(levels) - 1)
    )

    model.update()

    # Return variables and parameters for further analysis and/or plotting
    return (
        pv,
        load,
        unmet,
        curtail,
        pv_maxed_binary,
        total_load,
        penalty_cost,
        level_bin,
        levels,
        penalty_per_level,
        demand_level,
    )


### Case 3: EV, PV with Feed-in and Penalty implemented in the following function
def EV_PV_penalty_feed_in_case_3(
    Time_interval,
    merged_data,
    model,
    power_dishwasher,
    binary_dishwasher,
    power_wm,
    binary_wm,
    power_dryer,
    binary_dryer,
    kwh_per_km,
    inflexible_demand,
    max_power_hp,
    power_hp,
):
    """
    In this function EV charging and discharging with V2H and V2G, PV and Penalty are implemented.
    This function is only called in Case 3.
    """
    ### ---- EV CHARGING ---- ###

    # EV properties
    min_power_ev = 1  # Minimum power to charge the EV in kW
    max_power_ev = 10  # Maximum power to charge the EV in kW
    max_capacity_ev = 70  # Capacity of EV-Battery in kWh
    initial_soc_ev = 20  # Initial state of charge for EV battery in kWh

    # Variable: EV battery state of charge over time in kWh
    soc_ev = model.addVars(
        Time_interval, lb=0, ub=max_capacity_ev, vtype=GRB.CONTINUOUS, name="soc_ev"
    )

    # Variable: EV charging power in kW
    charging_ev = model.addVars(
        Time_interval,
        lb=0,
        ub=max_power_ev,
        vtype=GRB.CONTINUOUS,
        name="charging_lvl_ev",
    )

    # Variable: EV charging binary (1 if charging, 0 otherwise)
    binary_ev = model.addVars(Time_interval, vtype=GRB.BINARY, name="on_ev")

    # Constraint: EV SoC must be ≥80% by 7:00 each day
    for d in range(Time_interval // 24):
        t = d * 24 + 7  # 7:00 each day
        if t < Time_interval:
            model.addConstr(
                soc_ev[t] >= 0.8 * max_capacity_ev, name=f"ev_soc_7am_day_{d}"
            )

    # Constraint: EV can only charge if it's at home
    model.addConstrs(
        (
            binary_ev[t] <= merged_data["ev_at_home_binary"][t]
            for t in range(len(binary_ev))
        ),
        name="allowed_ev_summer",
    )

    # Constraint: Charging power upper and lower bounds using binary indicator
    model.addConstrs(
        (
            charging_ev[t]
            <= max_power_ev * merged_data["ev_home_availability"][t] * binary_ev[t]
            for t in range(Time_interval)
        ),
        name="max_power_ev",
    )
    model.addConstrs(
        (
            charging_ev[t]
            >= min_power_ev * merged_data["ev_home_availability"][t] * binary_ev[t]
            for t in range(Time_interval)
        ),
        name="min_power_ev",
    )

    # Constraint: Initial EV state of charge
    model.addConstr(soc_ev[0] == initial_soc_ev, name="ev_soc_initial")

    model.update()

    ### === Feed-in Tariff Setup === ###

    # Define the Feed-in Tariff as 80% of the Spotmarket at every timestep
    merged_data["feed_in_tariff"] = merged_data["Spotmarket_(EUR/kWh)"] * 0.8

    # Define upper and lower boundaries for the Feed-in Tariff
    merged_data["feed_in_tariff"] = merged_data["feed_in_tariff"].clip(
        lower=0.0, upper=0.2
    )

    min_disch_ev = min_power_ev  # Lower boundary for discharging = for charging
    max_disch_ev = max_power_ev  # Upper boundary for discharging = for charging

    # Variable: Total household load including EV, HP, appliances in kWh
    total_load = {
        t: merged_data["Inflexible_Demand_(kWh)"][t]
        + power_dishwasher * binary_dishwasher[t]
        + power_wm * binary_wm[t]
        + power_dryer * binary_dryer[t]
        + charging_ev[t]
        + power_hp[t]
        for t in range(Time_interval)
    }

    # Calculate demand levels for penalty tiering
    base_max_demand = (
        max(inflexible_demand)
        + power_dishwasher
        + power_wm
        + power_dryer
        + max_power_ev
    )

    max_demand = (
        max(inflexible_demand)
        + power_dishwasher
        + power_wm
        + power_dryer
        + max_power_ev
        + max_power_hp
    )

    ### === V2G / V2H Setup === ###

    # Variable: Binary, 1 if EV is discharging to house (V2H)
    ev_v2h_feed_in_binary = model.addVars(
        Time_interval, vtype=GRB.BINARY, name="ev_v2h_feed_in_binary"
    )

    # Variable: Binary, 1 if EV is discharging to grid (V2G)
    ev_feed_in_binary = model.addVars(
        Time_interval, vtype=GRB.BINARY, name="ev_feed_in_binary"
    )

    # Variable: EV discharge power to grid in kW
    ev_feed_in_power = model.addVars(Time_interval, lb=0.0, name="ev_feed_in_power")

    # Variable: EV discharge power to home (V2H) in kW
    ev_v2h_power = model.addVars(Time_interval, lb=0.0, name="ev_v2h_power")

    #  Constraint: EV cannot charge and discharge simultaneously
    model.addConstrs(
        (ev_v2h_feed_in_binary[t] + binary_ev[t] <= 1 for t in range(Time_interval)),
        name="ev_v2h_feed_in_binary_constraint",
    )
    model.addConstrs(
        (ev_feed_in_binary[t] + binary_ev[t] <= 1 for t in range(Time_interval)),
        name="ev_feed_in_binary_constraint",
    )

    # Constraint: Limit V2H power using binary and availability
    model.addConstrs(
        (
            ev_v2h_power[t]
            <= max_disch_ev
            * merged_data["ev_home_availability"][t]
            * ev_v2h_feed_in_binary[t]
            for t in range(Time_interval)
        ),
        name="max_power_ev_v2h_feed_in",
    )
    model.addConstrs(
        (
            ev_v2h_power[t]
            >= min_disch_ev
            * merged_data["ev_home_availability"][t]
            * ev_v2h_feed_in_binary[t]
            for t in range(Time_interval)
        ),
        name="min_power_ev_v2h_feed_in",
    )

    # Constraint: Limit V2G power using binary and availability
    model.addConstrs(
        (
            ev_feed_in_power[t]
            <= max_disch_ev
            * merged_data["ev_home_availability"][t]
            * ev_feed_in_binary[t]
            for t in range(Time_interval)
        ),
        name="max_power_ev_feed_in",
    )
    model.addConstrs(
        (
            ev_feed_in_power[t]
            >= min_disch_ev
            * merged_data["ev_home_availability"][t]
            * ev_feed_in_binary[t]
            for t in range(Time_interval)
        ),
        name="min_power_ev_feed_in",
    )

    # Constraint: Total discharge cannot exceed EV max power
    model.addConstrs(
        (
            ev_feed_in_power[t] + ev_v2h_power[t] <= max_disch_ev
            for t in range(Time_interval)
        ),
        name="ev_feed_in_power_limit",
    )

    # Constraint: EV SoC update with driving, charging, discharging
    model.addConstrs(
        (
            soc_ev[t]
            == soc_ev[t - 1]
            + charging_ev[t - 1]
            - ev_feed_in_power[t - 1]
            - ev_v2h_power[t - 1]
            - merged_data["distance_driven"][t - 1] * kwh_per_km
            for t in range(1, Time_interval)
        ),
        name="ev_soc_update",
    )

    ### === PV Production and Usage === ###

    # Variable: Binary, 1 if PV is insufficient for load
    pv_maxed_binary = model.addVars(Time_interval, vtype=GRB.BINARY, name="pv_maxed")

    # Variable: Unmet demand due to PV/EV shortfall in kWh
    unmet = model.addVars(Time_interval, lb=0.0, name="unmet_load")

    # Constraint: V2H power must not exceed unmet demand - beware: this constraint belongs to the V2H logic -> see mathematical formulation
    model.addConstrs(
        (
            ev_v2h_feed_in_binary[t] * ev_v2h_power[t] <= unmet[t]
            for t in range(Time_interval)
        ),
        name="ev_v2h_power_constraint",
    )

    # Variable: PV energy fed into the grid in kWh
    pv_feed_in = model.addVars(Time_interval, lb=0.0, name="feed_in")
    pv2H = model.addVars(Time_interval, lb=0.0, name="pv2H")

    # Constant: Large number for big-M method
    M = max(merged_data["PV_energy_production_kWh"]) + max_demand

    # Constraint: PV + V2H must cover load + feed-in + unmet
    for t in range(Time_interval):
        pv = merged_data["PV_energy_production_kWh"][t]
        load = total_load[t]

        # Constraint:PV2H must equal PV minus feed-in
        model.addConstr(pv2H[t] == pv - pv_feed_in[t])
        # Constraint: pv_feed in can't be bigger than PV production
        model.addConstr(pv_feed_in[t] <= pv)
        # Constraint: Energy balance at each timestep
        model.addConstr(
            pv + ev_v2h_power[t] - load + unmet[t] - pv_feed_in[t] == 0,
            name=f"pv_load_balance_{t}",
        )

        # Constraint: If PV exceeds load, PV feeds in
        model.addConstr(
            pv_feed_in[t] <= (1 - pv_maxed_binary[t]) * M, name=f"feed_in_pv_{t}_2"
        )

        # Constraint: If PV is less than load, we may have unmet
        model.addConstr(unmet[t] <= pv_maxed_binary[t] * M, name=f"unmet_load_{t}_2")

    ### === Unmet Load Penalty Logic === ###

    ε = 1e-3  # Small epsilon to avoid overlapping level ranges
    wanted_steps = 6  # Number of discrete penalty tiers you want to model

    levels = np.arange(
        0, base_max_demand / 7 * 6 + ε, base_max_demand / ((wanted_steps - 1))
    )
    levels = np.append(levels, max_demand + ε)

    # Multiplier: linear penalty per level
    multiplier_per_level = [0.003 * i for i in range(len(levels) - 1)]

    # Variable: Binary variable for which level is active
    level_bin = [
        [
            model.addVar(vtype=GRB.BINARY, name=f"level_bin[{t},{i}]")
            for i in range(len(levels) - 1)
        ]
        for t in range(Time_interval)
    ]

    # Variable: Integer demand level index (0–5)
    demand_level = [
        model.addVar(
            lb=0, ub=len(levels) - 1, vtype=GRB.INTEGER, name=f"demand_level[{t}]"
        )
        for t in range(Time_interval)
    ]

    # Constraint: Only one demand level active per timestep
    for t in range(Time_interval):
        model.addConstr(gp.quicksum(level_bin[t]) == 1, name=f"one_level_active_{t}")

    M_price = max_demand + 10  # Big-M for upper/lower bound constraints

    # Constraint: bind unmet_demand to its level using Big-M
    for t in range(Time_interval):
        for i in range(len(levels) - 1):
            # Constraint: Enforce level bounds using big-M
            model.addConstr(
                unmet[t] >= levels[i] - (1 - level_bin[t][i]) * M_price,
                name=f"lower_bound_level_{t}_{i}",
            )
            model.addConstr(
                unmet[t] <= levels[i + 1] - ε + (1 - level_bin[t][i]) * M_price,
                name=f"upper_bound_level_{t}_{i}",
            )
        # Constraint: Link demand level to binary
        model.addConstr(
            demand_level[t]
            == gp.quicksum(i * level_bin[t][i] for i in range(len(levels) - 1)),
            name=f"demand_level_calc_{t}",
        )

    # Variable: penalty_per_level[i] stores the penalty cost for demand level i, based on a linear penalty rate and the demand threshold for that level
    penalty_per_level = [
        multiplier_per_level[i] * levels[i] for i in range(len(levels) - 1)
    ]

    # Variable: Total penalty cost across all timesteps
    penalty_cost = gp.quicksum(
        penalty_per_level[i] * level_bin[t][i]
        for t in range(Time_interval)
        for i in range(len(levels) - 1)
    )

    model.update()

    # Return All Relevant Variables for objective function and plotting
    return (
        ev_feed_in_binary,
        ev_feed_in_power,
        ev_v2h_feed_in_binary,
        ev_v2h_power,
        pv_maxed_binary,
        unmet,
        pv_feed_in,
        penalty_cost,
        level_bin,
        levels,
        penalty_per_level,
        demand_level,
        total_load,
        charging_ev,
        soc_ev,
        binary_ev,
        pv2H,
    )
