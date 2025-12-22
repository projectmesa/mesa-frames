# Sugarscape IG benchmark assets

This folder contains **internal-only** benchmark tooling for Sugarscape IG.

- `numpy_engine.py`: a specialized NumPy/Numba step kernel for *benchmarking*.
- `sugarscape_ig_performance.md`: notes and measurement write-up.

These files are **not** part of the public mesa-frames example API.

## Running

For more representative timings, run with optimizations enabled so Python skips `__debug__` checks:

- `python -O benchmarks/sugarscape_ig/numpy_engine.py`

If you use `uv` for environment management, you can also do:

- `uv run python -O benchmarks/sugarscape_ig/numpy_engine.py`
