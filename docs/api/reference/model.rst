Model
=====

.. currentmodule:: mesa_frames

Quick intro
-----------

The ``Model`` orchestrates the simulation lifecycle: creating and registering ``AgentSet``s, stepping the simulation, and integrating with ``DataCollector`` and spatial ``Grid``s. Typical usage:

- Instantiate ``Model``, add ``AgentSet`` instances to ``model.sets``.
- Call ``model.sets.do('step')`` inside your model loop to trigger set-level updates.
- Use ``DataCollector`` to sample model- and agent-level columns each step.

Minimal example
---------------

.. code-block:: python

    from mesa_frames import Model, AgentSet, DataCollector
    import polars as pl

    class People(AgentSet):
        def step(self):
            self.add(pl.DataFrame({'wealth': [1, 2, 3]}))

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.sets += People(self)
            self.dc = DataCollector(model_reporters={'avg_wealth': lambda m: m.sets['People'].df['wealth'].mean()})

    m = MyModel()
    m.step()

API reference
-------------

.. tab-set::

    .. tab-item:: Overview

        .. rubric:: Lifecycle / Core

        .. autosummary::
            :nosignatures:
            :toctree:

            Model.__init__
            Model.step
            Model.run_model
            Model.reset_randomizer

        .. rubric:: Accessors / Properties

        .. autosummary::
            :nosignatures:
            :toctree:

            Model.steps
            Model.sets
            Model.space
            Model.seed

    .. tab-item:: Full API

        .. autoclass:: Model
            :autosummary:
            :autosummary-nosignatures:
