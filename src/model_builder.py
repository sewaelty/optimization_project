from gurobipy import Model, GRB

def build_model(data):
    model = Model("OptimizationModel")
    x = model.addVar(name="x", vtype=GRB.CONTINUOUS, lb=0)
    y = model.addVar(name="y", vtype=GRB.CONTINUOUS, lb=0)
    model.addConstr(x + y <= data["max"], name="c1")
    model.setObjective(2 * x + 3 * y, GRB.MAXIMIZE)
    return model
