import os
import sys

# Absolute path to project root (../../ from run.py)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Make sure Ray subprocesses also see this
os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")

import numpy as np
from deap import gp, tools
from flex.gp.regressor import GPSymbolicRegressor
import ray
from flex.gp import util
from flex.gp.primitives import add_primitives_to_pset_from_dict
from src.Fitness import fitness, assign_attributes
from src.PredictScoreFuncs import predict, score
import src.Functions
import time


# Define the numerical grid (Domain)
# 1. Generate initial ranges
x1_raw = np.linspace(-2, 2, 100)
x2_raw = np.linspace(-2, 2, 100)

# 2. Exclude/Clip the specific values from the axes
# This removes any individual coordinate that satisfies your logic
x1_vals = x1_raw[~(np.abs(x1_raw**2 - 1) < 0.02)]
x2_vals = x2_raw[~(x2_raw**2 < 0.02)]

# 3. Create meshgrid from the filtered axes
X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing="ij")


def write_topk_last_gen(gpsr, k=10, out_dir="top10_last_gen"):
    os.makedirs(out_dir, exist_ok=True)
    # final population across all islands
    pop = gpsr._GPSymbolicRegressor__flatten_list(gpsr._GPSymbolicRegressor__pop)
    topk = tools.selBest(pop, k=min(k, len(pop)))
    for rank, ind in enumerate(topk, 1):
        expr = getattr(ind, "expr", str(ind))   # falls back to str(ind) if not set
        consts = getattr(ind, "consts", None)
        fit = ind.fitness.values[0]
        with open(os.path.join(out_dir, f"ind_{rank}.txt"), "w") as f:
            f.write("Expression:\n")
            f.write(expr + "\n")
            f.write("Fitness:\n")
            f.write(f"{fit:.6g}\n")
            f.write("Parameters:\n")
            f.write(str(consts) + "\n")


def main():
    yamlfile = "config.yaml"
    filename = yamlfile

    regressor_params, config_file_data = util.load_config_data(filename)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [float, float],
        float,
    )

    pset.renameArguments(ARG0="x1", ARG1="x2")
    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])
    penalty = config_file_data["gp"]["penalty"]
    
    train_data = src.Functions.Dataset("true_data", [x1_vals, x2_vals], None)
    # attach grid ONCE
    train_data.X1 = X1
    train_data.X2 = X2
    train_data.grid_shape = X1.shape
    train_data.mesh = [X1, X2]
    
    common_data = {"true_data": train_data, "penalty": penalty}
    callback_func = assign_attributes
    pset.addTerminal(object, float, "a")
    
    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        callback_func=callback_func,
        print_log=True,
        batch_size=17,
        # seed_str=seed,
        **regressor_params
    )
    
    tic = time.time()
    gpsr.fit(train_data)
    toc = time.time()

    write_topk_last_gen(gpsr, k=100, out_dir="top10_last_gen")

    best_ind = gpsr.get_best_individuals(1)[0]   # Access the best individual

    best_parameters = getattr(best_ind, "consts", None)  # Save the best parameters
    if best_parameters is not None:
        print("Best parameters = ", best_parameters)

    print("Elapsed time = ", toc - tic)
    time_per_individual = (toc - tic) / (
        gpsr.generations * gpsr.num_individuals * gpsr.num_islands
    )
    print("Time per individual = ", time_per_individual)
    print("Individuals per sec = ", 1 / time_per_individual)

    # Access and save the best individual
    best_expression = str(best_ind)  # Save the best expression
    best_fitness = gpsr.get_train_fit_history()[-1]  # Save the best fitness score

    # Write best expression to a file
    with open("best_expression.txt", "w") as file:
        file.write(f"Best Expression:\n{best_expression}\n")
        file.write(f"Best Fitness:\n{best_fitness}\n")
        file.write(f"Best Parameters:\n{best_parameters}\n")

    print("Best Expression and Fitness saved to 'best_expression.txt'.")
    # print("Best Expression Sympy", str(gpsr.get_best_individual_sympy()))

    ray.shutdown()


if __name__ == "__main__":
    main()
