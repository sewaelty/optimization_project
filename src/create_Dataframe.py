def createDataframe(season):
    """Create a dataframe for the given season.
    Args:
        season (str): The season for which to return the database.
    """

    ### imports

    import os
    import pandas as pd

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")

    ### Load prices

    # Summer month
    p_summer = pd.read_csv(
        os.path.join(data_dir, "Spotmarket_August_Corrected.csv"), sep=","
    )
    # change price from euro/mwh to euro/kwh and renaming the column
    p_summer["price_EUR_MWh"] = p_summer["price_EUR_MWh"] / 1000
    p_summer.rename(columns={"price_EUR_MWh": "Spotmarket_(EUR/kWh)"}, inplace=True)

    # Winter month
    p_winter = pd.read_csv(
        os.path.join(data_dir, "Spotmarket_December_Corrected.csv"), sep=","
    )
    # change price from euro/mwh to euro/kwh and renaming the column
    p_winter["price_EUR_MWh"] = p_winter["price_EUR_MWh"] / 1000
    p_winter.rename(columns={"price_EUR_MWh": "Spotmarket_(EUR/kWh)"}, inplace=True)

    # Spotmarket data from: https://energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=CH&interval=month&year=2024&legendItems=by4&month=12

    ### Load fixed appliances

    # TV consumption data for summer month
    tv_summer = pd.read_csv(
        os.path.join(data_dir, "tv_consumption_august_2024_detailed.csv"), sep=","
    )

    # TV consumption data for winter month
    tv_winter = pd.read_csv(
        os.path.join(data_dir, "tv_consumption_december_2024_detailed.csv"), sep=","
    )

    # Lighting consumption data for summer month
    lighting_summer = pd.read_csv(
        os.path.join(data_dir, "lighting_consumption_august_2024.csv"), sep=","
    )

    # Lighting consumption data for winter month
    lighting_winter = pd.read_csv(
        os.path.join(data_dir, "lighting_consumption_december_2024.csv"), sep=","
    )

    # Fridge consumption data for summer month
    fridge_summer = pd.read_csv(
        os.path.join(data_dir, "fridge_August_Final_Adjusted.csv"), sep=","
    )

    # Fridge consumption data for winter month
    fridge_winter = pd.read_csv(
        os.path.join(data_dir, "fridge_December_Final_Adjusted.csv"), sep=","
    )

    # Oven consumption data for summer month
    oven_summer = pd.read_csv(
        os.path.join(data_dir, "Oven_Energy_Consumption_August_Final.csv"), sep=","
    )

    # Oven consumption data for winter month
    oven_winter = pd.read_csv(
        os.path.join(data_dir, "Oven_Energy_Consumption_December_Final.csv"), sep=","
    )

    # Induction stove consumption data for summer month
    induction_summer = pd.read_csv(
        os.path.join(data_dir, "Induction_Stove_Energy_Consumption_August_Final.csv"),
        sep=",",
    )

    # Induction stove consumption data for winter month
    induction_winter = pd.read_csv(
        os.path.join(data_dir, "Induction_Stove_Energy_Consumption_December_Final.csv"),
        sep=",",
    )

    # adjust names of columns for summer
    tv_summer.columns = tv_summer.columns.str.replace(
        "tv_power_kWh", "TV_Consumption_(kWh)"
    )
    lighting_summer.columns = lighting_summer.columns.str.replace(
        "lighting_power_kWh", "Lighting_Consumption_(kWh)"
    )
    fridge_summer.columns = fridge_summer.columns.str.replace(
        "consumption_kWh", "Fridges_Consumption_(kWh)"
    )
    oven_summer.columns = oven_summer.columns.str.replace(
        "consumption_kWh", "Oven_Consumption_(kWh)"
    )
    induction_summer.columns = induction_summer.columns.str.replace(
        "consumption_kWh", "Induction_Stove_Consumption_(kWh)"
    )

    # adjust names of columns for winter
    tv_winter.columns = tv_winter.columns.str.replace(
        "tv_power_kWh", "TV_Consumption_(kWh)"
    )
    lighting_winter.columns = lighting_winter.columns.str.replace(
        "lighting_power_kWh", "Lighting_Consumption_(kWh)"
    )
    fridge_winter.columns = fridge_winter.columns.str.replace(
        "consumption_kWh", "Fridges_Consumption_(kWh)"
    )
    oven_winter.columns = oven_winter.columns.str.replace(
        "consumption_kWh", "Oven_Consumption_(kWh)"
    )
    induction_winter.columns = induction_winter.columns.str.replace(
        "consumption_kWh", "Induction_Stove_Consumption_(kWh)"
    )

    ### Load PV data

    # PV data for summer month
    pv_summer_total = pd.read_csv(os.path.join(data_dir, "pv_august.csv"), sep=",")
    pv_summer_total["PV_energy_production_kWh"] = pv_summer_total[
        "PV_energy_production_kWh"
    ]

    # PV data for winter month
    pv_winter_total = pd.read_csv(os.path.join(data_dir, "pv_december.csv"), sep=",")
    pv_winter_total["PV_energy_production_kWh"] = pv_winter_total[
        "PV_energy_production_kWh"
    ]

    # extract the timestamp and the Energy_production columns $
    pv_summer = pv_summer_total[["timestamp", "PV_energy_production_kWh"]]
    pv_winter = pv_winter_total[["timestamp", "PV_energy_production_kWh"]]

    ### Build Database

    # Ensure all timestamp columns are of the same type for summer
    p_summer["timestamp"] = pd.to_datetime(p_summer["timestamp"])
    lighting_summer["timestamp"] = pd.to_datetime(lighting_summer["timestamp"])
    fridge_summer["timestamp"] = pd.to_datetime(fridge_summer["timestamp"])
    oven_summer["timestamp"] = pd.to_datetime(oven_summer["timestamp"])
    induction_summer["timestamp"] = pd.to_datetime(induction_summer["timestamp"])
    tv_summer["timestamp"] = pd.to_datetime(tv_summer["timestamp"])
    pv_summer["timestamp"] = pd.to_datetime(
        pv_summer["timestamp"].copy(), format="%Y-%m-%d %H:%M:%S"
    )  # -> already done in the previous step

    # Ensure all timestamp columns are of the same type for winter
    p_winter["timestamp"] = pd.to_datetime(p_winter["timestamp"])
    lighting_winter["timestamp"] = pd.to_datetime(lighting_winter["timestamp"])
    fridge_winter["timestamp"] = pd.to_datetime(fridge_winter["timestamp"])
    oven_winter["timestamp"] = pd.to_datetime(oven_winter["timestamp"])
    induction_winter["timestamp"] = pd.to_datetime(induction_winter["timestamp"])
    tv_winter["timestamp"] = pd.to_datetime(tv_winter["timestamp"])
    pv_winter["timestamp"] = pd.to_datetime(
        pv_winter["timestamp"].copy(), format="%Y-%m-%d %H:%M:%S"
    )  # -> already done in the previous step

    # change year from 2024 to 2023 for summer data in timestamp
    tv_summer["timestamp"] = tv_summer["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    lighting_summer["timestamp"] = lighting_summer["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    fridge_summer["timestamp"] = fridge_summer["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    oven_summer["timestamp"] = oven_summer["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    induction_summer["timestamp"] = induction_summer["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )

    # change year from 2024 to 2023 for winter data
    tv_winter["timestamp"] = tv_winter["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    lighting_winter["timestamp"] = lighting_winter["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    fridge_winter["timestamp"] = fridge_winter["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    oven_winter["timestamp"] = oven_winter["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )
    induction_winter["timestamp"] = induction_winter["timestamp"].apply(
        lambda x: x.replace(year=2023)
    )

    # shorten the dataset to 4 weeks (4 weeks * 7 days * 24 hours = 672 rows) for summer
    lighting_summer = lighting_summer.iloc[: 4 * 7 * 24]
    fridge_summer = fridge_summer.iloc[: 4 * 7 * 24]
    oven_summer = oven_summer.iloc[: 4 * 7 * 24]
    induction_summer = induction_summer.iloc[: 4 * 7 * 24]
    tv_summer = tv_summer.iloc[: 4 * 7 * 24]
    p_summer = p_summer.iloc[: 4 * 7 * 24]
    pv_summer = pv_summer.iloc[: 4 * 7 * 24]

    # shorten the dataset to 4 weeks (4 weeks * 7 days * 24 hours = 672 rows) for winter
    lighting_winter = lighting_winter.iloc[: 4 * 7 * 24]
    fridge_winter = fridge_winter.iloc[: 4 * 7 * 24]
    oven_winter = oven_winter.iloc[: 4 * 7 * 24]
    induction_winter = induction_winter.iloc[: 4 * 7 * 24]
    tv_winter = tv_winter.iloc[: 4 * 7 * 24]
    p_winter = p_winter.iloc[: 4 * 7 * 24]
    pv_winter = pv_winter.iloc[: 4 * 7 * 24]

    # Adjust the timestamp for fridge
    fridge_summer["timestamp"] = fridge_summer["timestamp"] - pd.Timedelta(hours=1)
    fridge_winter["timestamp"] = fridge_winter["timestamp"] - pd.Timedelta(hours=1)

    # Sum up inflexible demand: lighting, tv, fridge, oven, induction stove
    inflexible_demand_summer = pd.DataFrame()
    inflexible_demand_summer["timestamp"] = lighting_summer["timestamp"]
    inflexible_demand_summer["Inflexible_Demand_(kWh)"] = (
        lighting_summer["Lighting_Consumption_(kWh)"]
        + tv_summer["TV_Consumption_(kWh)"]
        + fridge_summer["Fridges_Consumption_(kWh)"]
        + oven_summer["Oven_Consumption_(kWh)"]
        + induction_summer["Induction_Stove_Consumption_(kWh)"]
    )

    inflexible_demand_winter = pd.DataFrame()
    inflexible_demand_winter["timestamp"] = lighting_winter["timestamp"]
    inflexible_demand_winter["Inflexible_Demand_(kWh)"] = (
        lighting_winter["Lighting_Consumption_(kWh)"]
        + tv_winter["TV_Consumption_(kWh)"]
        + fridge_winter["Fridges_Consumption_(kWh)"]
        + oven_winter["Oven_Consumption_(kWh)"]
        + induction_winter["Induction_Stove_Consumption_(kWh)"]
    )
    
    ev_data_summer = pd.read_csv(os.path.join(data_dir, "ev_data_hourly_5weeks_summer_2023.csv"), sep=",")
    ev_data_winter = pd.read_csv(os.path.join(data_dir, "ev_data_hourly_5weeks_winter_2023.csv"), sep=",")
    # Ensure all timestamp columns are of the same type for summer
    ev_data_summer["timestamp"] = pd.to_datetime(ev_data_summer["datetime"])
    ev_data_winter["timestamp"] = pd.to_datetime(ev_data_winter["datetime"])
    #drop column datetime from ev_data_summer and ev_data_winter
    ev_data_summer.drop(columns=["datetime"], inplace=True)
    ev_data_winter.drop(columns=["datetime"], inplace=True)
    #shorten both datasets to 4 weeks (4 weeks * 7 days * 24 hours = 672 rows)
    
    ev_data_summer = ev_data_summer.iloc[: 4 * 7 * 24]
    ev_data_winter = ev_data_winter.iloc[: 4 * 7 * 24]
    
    # Merge all datasets on the 'timestamp' column for summer
    merged_data_summer = p_summer.merge(
        inflexible_demand_summer, left_on="timestamp", right_on="timestamp", how="inner"
    ).merge(pv_summer, left_on="timestamp", right_on="timestamp", how="inner")

    # Merge all datasets on the 'timestamp' column for winter
    merged_data_winter = p_winter.merge(
        inflexible_demand_winter, left_on="timestamp", right_on="timestamp", how="inner"
    ).merge(pv_winter, left_on="timestamp", right_on="timestamp", how="inner")

    heat_demand_winter = pd.read_csv(
        os.path.join(data_dir, "heating_demand_december.csv"), sep=","
    )
    merged_data_winter["Heating_Demand_(kWh)"] = heat_demand_winter[
        "Hot water + Space Heating demand [kWh]"
    ]
    
    #merge ev data to merged_data_summer and merged_data_winter
    merged_data_summer = merged_data_summer.merge(
        ev_data_summer, left_on="timestamp", right_on="timestamp", how="inner"
    )
    merged_data_winter = merged_data_winter.merge(
        ev_data_winter, left_on="timestamp", right_on="timestamp", how="inner"
    )

    # return the desired datamframe
    if season == "summer":
        return  merged_data_summer
    elif season == "winter":
        return merged_data_winter
    else:
        raise ValueError("Season must be either 'summer' or 'winter'.")
