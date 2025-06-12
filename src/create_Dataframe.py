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
    # replace 0 values with 0.01
    p_winter["Spotmarket_(EUR/kWh)"] = (
        p_winter["Spotmarket_(EUR/kWh)"].copy().replace(0, 0.01)
    )

    # Spotmarket data from: https://energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=CH&interval=month&year=2024&legendItems=by4&month=12

    ### Load fixed appliances
    appliances_winter = pd.read_csv(
        os.path.join(data_dir, "DeviceProfiles_3600s.Electricity_december.csv"), sep=";"
    )

    fixed_appliances_winter = appliances_winter.drop(
        [
            "Electricity.Timestep",
            "HH1 - Kitchen - Dishwasher NEFF SD6P1F (2011) [kWh]",
            "HH1 - Kitchen - Dryer / Miele T 8626 WP [kWh]",
            "HH1 - Kitchen - Washing Machine AEG Öko Plus 1400 [kWh]",
        ],
        axis=1,
    )

    fixed_appliances_winter.columns = fixed_appliances_winter.columns.str.replace(
        "Time", "timestamp"
    )
    fixed_appliances_winter["timestamp"] = pd.to_datetime(
        fixed_appliances_winter["timestamp"], dayfirst=True
    )

    ### loading summer appliance data

    appliances_summer = pd.read_csv(
        os.path.join(data_dir, "DeviceProfiles_3600s.Electricity_august.csv"), sep=";"
    )

    fixed_appliances_summer = appliances_summer.drop(
        [
            "Electricity.Timestep",
            "HH1 - Kitchen - Dishwasher NEFF SD6P1F (2011) [kWh]",
            "HH1 - Kitchen - Dryer / Miele T 8626 WP [kWh]",
            "HH1 - Kitchen - Washing Machine AEG Öko Plus 1400 [kWh]",
        ],
        axis=1,
    )

    fixed_appliances_summer.columns = fixed_appliances_summer.columns.str.replace(
        "Time", "timestamp"
    )

    fixed_appliances_summer["timestamp"] = pd.to_datetime(
        fixed_appliances_summer["timestamp"], dayfirst=True
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

    pv_summer["timestamp"] = pd.to_datetime(
        pv_summer["timestamp"].copy(), format="%Y-%m-%d %H:%M:%S"
    )  # -> already done in the previous step

    # Ensure all timestamp columns are of the same type for winter
    p_winter["timestamp"] = pd.to_datetime(p_winter["timestamp"])

    pv_winter["timestamp"] = pd.to_datetime(
        pv_winter["timestamp"].copy(), format="%Y-%m-%d %H:%M:%S"
    )  # -> already done in the previous step

    # Sum up inflexible demand
    inflexible_demand_summer = pd.DataFrame()
    inflexible_demand_summer["timestamp"] = fixed_appliances_summer["timestamp"]
    inflexible_demand_summer["Inflexible_Demand_(kWh)"] = fixed_appliances_summer.sum(
        axis=1, numeric_only=True
    )

    inflexible_demand_winter = pd.DataFrame()
    inflexible_demand_winter["timestamp"] = fixed_appliances_winter["timestamp"]
    inflexible_demand_winter["Inflexible_Demand_(kWh)"] = fixed_appliances_winter.sum(
        axis=1, numeric_only=True
    )

    ev_data_summer = pd.read_csv(
        os.path.join(data_dir, "ev_data_hourly_5weeks_summer_2023.csv"), sep=","
    )
    ev_data_winter = pd.read_csv(
        os.path.join(data_dir, "ev_data_hourly_5weeks_winter_2023.csv"), sep=","
    )
    # Ensure all timestamp columns are of the same type for summer
    ev_data_summer["timestamp"] = pd.to_datetime(ev_data_summer["datetime"])
    ev_data_winter["timestamp"] = pd.to_datetime(ev_data_winter["datetime"])
    # drop column datetime from ev_data_summer and ev_data_winter
    ev_data_summer.drop(columns=["datetime"], inplace=True)
    ev_data_winter.drop(columns=["datetime"], inplace=True)

    # shorten both datasets to exactly respective month (august or december)
    ev_data_summer = ev_data_summer.iloc[: 31 * 24]
    ev_data_winter = ev_data_winter.iloc[: 31 * 24]

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

    # merge ev data to merged_data_summer and merged_data_winter
    merged_data_summer = merged_data_summer.merge(
        ev_data_summer, left_on="timestamp", right_on="timestamp", how="inner"
    )
    # merged_data_summer = merged_data_summer.iloc[: 7 * 4 * 24]  # only keep the first 7 days of data

    merged_data_winter = merged_data_winter.merge(
        ev_data_winter, left_on="timestamp", right_on="timestamp", how="inner"
    )
    # merged_data_winter = merged_data_winter.iloc[: 7 * 4 * 24]  # only keep the first 7 days of data

    # return the desired datamframe
    if season == "summer":
        return merged_data_summer[: 28 * 24]
    elif season == "winter":
        return merged_data_winter[: 28 * 24]
    else:
        raise ValueError("Season must be either 'summer' or 'winter'.")
