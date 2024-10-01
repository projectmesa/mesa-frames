import math

import matplotlib.pyplot as plt
import numpy as np
import perfplot
import polars as pl
import seaborn as sns
from ss_mesa.model import SugarscapeMesa
from ss_pandas.model import SugarscapePandas
from ss_polars.agents import (
    AntPolarsLoopDF,
    AntPolarsLoopNoVec,
    AntPolarsNumbaCPU,
    AntPolarsNumbaGPU,
    AntPolarsNumbaParallel,
)
from ss_polars.model import SugarscapePolars


class SugarScapeSetup:
    def __init__(self, n: int):
        if n >= 10**6:
            density = 0.17  # FLAME2-GPU
        else:
            density = 0.04  # mesa
        self.n = n
        self.seed = 42
        dimension = math.ceil(math.sqrt(n / density))
        random_gen = np.random.default_rng(self.seed)
        self.sugar_grid = random_gen.integers(0, 4, (dimension, dimension))
        self.initial_sugar = random_gen.integers(6, 25, n)
        self.metabolism = random_gen.integers(2, 4, n)
        self.vision = random_gen.integers(1, 6, n)
        self.initial_positions = pl.DataFrame(
            schema={"dim_0": pl.Int64, "dim_1": pl.Int64}
        )
        while self.initial_positions.shape[0] < n:
            initial_pos_0 = random_gen.integers(
                0, dimension, n - self.initial_positions.shape[0]
            )
            initial_pos_1 = random_gen.integers(
                0, dimension, n - self.initial_positions.shape[0]
            )
            self.initial_positions = self.initial_positions.vstack(
                pl.DataFrame(
                    {
                        "dim_0": initial_pos_0,
                        "dim_1": initial_pos_1,
                    }
                )
            ).unique(maintain_order=True)
        return


def mesa_implementation(setup: SugarScapeSetup):
    model = SugarscapeMesa(
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.initial_positions,
        setup.seed,
    )
    model.run_model(100)
    return model


def mesa_frames_pandas_concise(setup: SugarScapeSetup):
    return SugarscapePandas(
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.seed,
    ).run_model(100)


def mesa_frames_polars_loop_DF(setup: SugarScapeSetup):
    model = SugarscapePolars(
        AntPolarsLoopDF,
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.initial_positions,
        setup.seed,
    )
    model.run_model(100)


def mesa_frames_polars_loop_no_vec(setup: SugarScapeSetup):
    model = SugarscapePolars(
        AntPolarsLoopNoVec,
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.initial_positions,
        setup.seed,
    )
    model.run_model(100)


def mesa_frames_polars_numba_cpu(setup: SugarScapeSetup):
    model = SugarscapePolars(
        AntPolarsNumbaCPU,
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.initial_positions,
        setup.seed,
    )
    model.run_model(100)


def mesa_frames_polars_numba_gpu(setup: SugarScapeSetup):
    model = SugarscapePolars(
        AntPolarsNumbaGPU,
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.initial_positions,
        setup.seed,
    )
    model.run_model(100)


def mesa_frames_polars_numba_parallel(setup: SugarScapeSetup):
    model = SugarscapePolars(
        AntPolarsNumbaParallel,
        setup.n,
        setup.sugar_grid,
        setup.initial_sugar,
        setup.metabolism,
        setup.vision,
        setup.initial_positions,
        setup.seed,
    )
    model.run_model(100)


def plot_and_print_benchmark(labels, kernels, n_range, title, image_path):
    out = perfplot.bench(
        setup=SugarScapeSetup,
        kernels=kernels,
        labels=labels,
        n_range=n_range,
        xlabel="Number of agents",
        equality_check=None,
        title=title,
    )
    plt.ylabel("Execution time (s)")
    out.save(image_path)
    print("\nExecution times:")
    for i, label in enumerate(labels):
        print(f"---------------\n{label}:")
        for n, t in zip(out.n_range, out.timings_s[i]):
            print(f"  Number of agents: {n}, Time: {t:.2f} seconds")
        print("---------------")


def main():
    # Mesa comparison
    sns.set_theme(style="whitegrid")
    """labels_0 = [
        "mesa",
        # "mesa-frames (pd concise)",
        "mesa-frames (pl concise)",
    ]
    kernels_0 = [
        mesa_implementation,
        # mesa_frames_pandas_concise,
        mesa_frames_polars_concise,
    ]
    n_range_0 = [k for k in range(1, 100002, 10000)]
    title_0 = "100 steps of the SugarScape IG model:\n" + " vs ".join(labels_0)
    image_path_0 = "benchmark_plot_0.png"
    plot_and_print_benchmark(labels_0, kernels_0, n_range_0, title_0, image_path_0)"""

    # FLAME2-GPU comparison
    labels_1 = [
        # "mesa-frames (pd concise)",
        "mesa-frames (pl loop DF)",
        "mesa-frames (pl loop no vec)",
        "mesa-frames (pl numba CPU)",
        "mesa-frames (pl numba parallel)",
        # "mesa-frames (pl numba GPU)",
    ]
    # Polars best_moves (non-vectorized loop vs DF loop vs numba loop)
    kernels_1 = [
        # mesa_frames_pandas_concise,
        mesa_frames_polars_loop_DF,
        mesa_frames_polars_loop_no_vec,
        mesa_frames_polars_numba_cpu,
        mesa_frames_polars_numba_parallel,
        # mesa_frames_polars_numba_gpu,
    ]
    n_range_1 = [k for k in range(1, 2 * 10**6 + 2, 10**6)]
    # n_range_1 = [k for k in range(10000, 100002, 10000)]
    title_1 = "100 steps of the SugarScape IG model:\n" + " vs ".join(labels_1)
    image_path_1 = "polars_comparison.png"
    plot_and_print_benchmark(labels_1, kernels_1, n_range_1, title_1, image_path_1)


if __name__ == "__main__":
    main()
