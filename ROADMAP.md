# Roadmap

This document outlines the near-term roadmap for mesa-frames as of October 2025.

## 1) LazyFrames for Polars + GPU

Switch Polars usage from eager to `LazyFrame` to enable better query optimization and GPU acceleration.

Related issues:

- [#52: Use of LazyFrames for Polars implementation](https://github.com/mesa/mesa-frames/issues/52)

- [#144: Switch to LazyFrame for Polars implementation (PR)](https://github.com/mesa/mesa-frames/pull/144)

- [#89: Investigate Ibis or Narwhals for backend flexibility](https://github.com/mesa/mesa-frames/issues/89)

- [#122: Deprecate DataFrameMixin (remove during LazyFrames refactor)](https://github.com/mesa/mesa-frames/issues/122)

Progress and next steps:

- Land [#144](https://github.com/mesa/mesa-frames/pull/144) and convert remaining eager paths to lazy.

- Validate GPU execution paths and benchmark improvements.

- Revisit Ibis/Narwhals after LazyFrame stabilization.

- Fold DataFrameMixin removal into the LazyFrames transition ([#122](https://github.com/mesa/mesa-frames/issues/122)).

---

## 2) AgentSet Enhancements

Expose movement methods from `AgentContainer` and provide optimized utilities for "move to optimal" workflows.

Related issues:

- [#108: Adding abstraction of optimal agent movement](https://github.com/mesa/mesa-frames/issues/108)

- [#118: Adds move_to_optimal in DiscreteSpaceDF (PR)](https://github.com/mesa/mesa-frames/pull/118)

- [#82: Add movement methods to AgentContainer](https://github.com/mesa/mesa-frames/issues/82)

Next steps:

- Consolidate movement APIs under `AgentContainer`.

- Keep conflict resolution simple, vectorized, and well-documented.

---

## 3) Research & Publication

JOSS paper preparation and submission.

Related items:

- [#90: JOSS paper for the package](https://github.com/mesa/mesa-frames/issues/90)

- [#107: paper - Adding Statement of Need (PR)](https://github.com/mesa/mesa-frames/pull/107)

---

See [our contribution guide](/mesa-frames/contributing/) and browse all open items at <https://github.com/mesa/mesa-frames/issues>
