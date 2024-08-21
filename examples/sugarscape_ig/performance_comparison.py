import math

import matplotlib.pyplot as plt
import numpy as np
import perfplot
from ss_mesa.model import SugarscapeMesa
from ss_pandas.model import SugarscapePandas
from ss_polars.model import SugarscapePolars


class SugarScapeSetup:
    def __init__(self, n: int):
        if n >= 10**6:
            density = 0.17  # FLAME2-GPU
        else:
            density = 0.04  # mesa
        self.n = n
        dimension = math.ceil(math.sqrt(n / density))
        self.sugar_grid = np.random.randint(0, 4, (dimension, dimension))
        self.initial_sugar = np.random.randint(6, 25, n)
        self.metabolism = np.random.randint(2, 4, n)
        self.vision = np.random.randint(1, 6, n)


def mesa_implementation(setup: SugarScapeSetup):
    return SugarscapeMesa(
        setup.n, setup.sugar_grid, setup.initial_sugar, setup.metabolism, setup.vision
    ).run_model(100)


def mesa_frames_pandas_concise(setup: SugarScapeSetup):
    return SugarscapePandas(
        setup.n, setup.sugar_grid, setup.initial_sugar, setup.metabolism, setup.vision
    ).run_model(100)


def mesa_frames_polars_concise(setup: SugarScapeSetup):
    return SugarscapePolars(
        setup.n, setup.sugar_grid, setup.initial_sugar, setup.metabolism, setup.vision
    ).run_model(100)


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
    """# Mesa comparison
    sns.set_theme(style="whitegrid")
    labels_0 = [
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
        "mesa-frames (pl concise)",
    ]
    kernels_1 = [
        # mesa_frames_pandas_concise,
        mesa_frames_polars_concise,
    ]
    n_range_1 = [k for k in range(1, 3 * 10**6 + 2, 10**6)]
    title_1 = "100 steps of the SugarScape IG model:\n" + " vs ".join(labels_1)
    image_path_1 = "benchmark_plot_1.png"
    plot_and_print_benchmark(labels_1, kernels_1, n_range_1, title_1, image_path_1)


if __name__ == "__main__":
    main()
