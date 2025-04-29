from src.data_loader import load_data
from src.model_builder import build_model
from src.optimizer import solve_model

def main():
    data = load_data("data/input_data.csv")
    model = build_model(data)
    solve_model(model)

if __name__ == "__main__":
    main()
