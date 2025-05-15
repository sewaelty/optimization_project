from gurobipy import Model, GRB


def build_model(data):
    model = Model("OptimizationModel")
    # Inflexible variables
    """lights = model.addVar(name="lights", vtype=GRB.BINARY, lb=0, ub=1)
    fridge = model.addVar(name="fridge", vtype=GRB.BINARY, lb=0, ub=1)
    tv = model.addVar(name="tv", vtype=GRB.BINARY, lb=0, ub=1)
    stove = model.addVar(name="stove", vtype=GRB.BINARY, lb=0, ub=1)"""
    # Fixed flexible variables
    """washing_machine = model.addVar(name="washing_machine", vtype=GRB.BINARY, lb=0, ub=1)
    tumbler_dryer = model.addVar(name="tumbler_dryer", vtype=GRB.BINARY, lb=0, ub=1)
    dishwasher = model.addVar(name="dishwasher", vtype=GRB.BINARY, lb=0, ub=1)"""
    # Heatpump

    # Electriv Vehicle

    x = model.addVar(name="x", vtype=GRB.CONTINUOUS, lb=0)
    y = model.addVar(name="y", vtype=GRB.CONTINUOUS, lb=0)
    model.addConstr(x + y <= data["max"], name="c1")
    model.setObjective(2 * x + 3 * y, GRB.MAXIMIZE)
    return model
