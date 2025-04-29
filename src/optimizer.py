def solve_model(model):
    model.optimize()
    if model.status == 2:  # Optimal
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")
        print(f"Objective: {model.ObjVal}")
    else:
        print("No optimal solution found.")
