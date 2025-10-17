Data Collection
===============

.. currentmodule:: mesa_frames

Quick intro
-----------

``DataCollector`` samples model- and agent-level columns over time and returns cleaned DataFrames suitable for analysis. Typical patterns:

- Provide ``model_reporters`` (callables producing scalars) and ``agent_reporters`` (column selectors or callables that operate on an AgentSet).
- Call ``collector.collect(model)`` inside the model step or use built-in integration if the model calls the collector automatically.

Minimal example
---------------

.. code-block:: python

    from mesa_frames import DataCollector, Model, AgentSet
    import polars as pl

    class P(AgentSet):
        def __init__(self, model):
            super().__init__(model)
            self.add(pl.DataFrame({'x': [1,2]}))

    class M(Model):
        def __init__(self):
            super().__init__()
            self.sets += P(self)
            self.dc = DataCollector(model_reporters={'count': lambda m: len(m.sets['P'])},
                                    agent_reporters='x')

    m = M()
    m.dc.collect()

API reference
-------------

.. tab-set::

    .. tab-item:: Overview

        .. rubric:: Lifecycle / Core

        .. autosummary::
            :nosignatures:
            :toctree:

            DataCollector.__init__
            DataCollector.collect
            DataCollector.conditional_collect
            DataCollector.flush
            DataCollector.data

        .. rubric:: Reporting / Internals

        .. autosummary::
            :nosignatures:
            :toctree:

            DataCollector.seed

    .. tab-item:: Full API

        .. autoclass:: DataCollector
            :autosummary:
            :autosummary-nosignatures:
