# Benchmarking in mesa-frames

As a library focused on performance improvements, it's crucial that mesa-frames maintains its speed advantages over time. To ensure this, we've implemented an automated benchmarking system that runs on every pull request targeting the main branch.

## How the Benchmark Workflow Works

The automated benchmark workflow runs on GitHub Actions and performs the following steps:

1. Sets up a Python environment with all necessary dependencies
2. Installs optional GPU dependencies (if available in the runner)
3. Runs a small subset of our benchmark examples:
   - SugarScape model (with 50,000 agents)
   - Boltzmann Wealth model (with 10,000 agents)
4. Generates timing results comparing mesa-frames to the original Mesa implementation
5. Produces a visualization of the benchmark results
6. Posts a comment on the PR with the benchmark results
7. Uploads full benchmark artifacts for detailed inspection

## Interpreting Benchmark Results

When reviewing a PR with benchmark results, look for:

1. **Successful execution**: The benchmarks should complete without errors
2. **Performance impact**: Check if the PR introduces any performance regressions
3. **Expected changes**: If the PR is aimed at improving performance, verify that the benchmarks show the expected improvements

The benchmark comment will include:

- Execution time for both mesa-frames and Mesa implementations
- The speedup factor (how many times faster mesa-frames is compared to Mesa)
- A visualization comparing the performance

## Running Benchmarks Locally

To run the same benchmarks locally and compare your changes to the current main branch:

```bash
# Clone the repository
git clone https://github.com/projectmesa/mesa-frames.git
cd mesa-frames

# Install dependencies
pip install -e ".[dev]"
pip install perfplot matplotlib seaborn

# Run the Sugarscape benchmark
cd examples/sugarscape_ig
python performance_comparison.py

# Run the Boltzmann Wealth benchmark
cd ../boltzmann_wealth
python performance_plot.py
```

The full benchmarks will take longer to run than the CI version as they test with more agents.

## Adding New Benchmarks

When adding new models or features to mesa-frames, consider adding benchmark tests to ensure their performance:

1. Create a benchmark script in the `examples` directory
2. Implement both mesa-frames and Mesa versions of the model
3. Use the `perfplot` library to measure and visualize performance
4. Update the GitHub Actions workflow to include your new benchmark (with a small dataset for CI)

## Tips for Performance Optimization

When optimizing code in mesa-frames:

1. **Always benchmark your changes**: Don't assume changes will improve performance without measuring
2. **Focus on real-world use cases**: Optimize for patterns that users are likely to encounter
3. **Balance readability and performance**: Code should remain maintainable even while being optimized
4. **Document performance characteristics**: Note any trade-offs or specific usage patterns that affect performance
5. **Test on different hardware**: If possible, verify improvements on both CPU and GPU environments

Remember that consistent, predictable performance is often more valuable than squeezing out every last bit of speed at the cost of complexity or stability.
