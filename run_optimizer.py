from optimizer import run_nsga2

run_nsga2(
    n_max=60,
    pop_size=20,
    n_gen=8,
    seed=42,
    out_dir="outputs/nsga2_test",
)

