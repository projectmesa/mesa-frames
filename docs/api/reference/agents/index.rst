Agents
======

.. currentmodule:: mesa_frames

Quick intro
-----------

- ``AgentSet`` stores agents as rows in a Polars-backed table and provides vectorised operations for high-performance updates. 

- ``AgentSetRegistry`` (available at ``model.sets``) is the container that holds all ``AgentSet`` instances for a model and provides convenience operations (add/remove sets, step all sets, rename). 

- Keep agent logic column-oriented and prefer Polars expressions for updates.

Minimal example
---------------

.. code-block:: python

    from mesa_frames import Model, AgentSet
    import polars as pl

    class MySet(AgentSet):
        def __init__(self, model):
            super().__init__(model)
            self.add(pl.DataFrame({"age": [0, 5, 10]}))

        def step(self):
            # vectorised update: increase age for all agents
            self.df = self.df.with_columns((pl.col("age") + 1).alias("age"))

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            # register an AgentSet on the model's registry
            self.sets += MySet(self)

    m = MyModel()
    # step all registered sets (delegates to each AgentSet.step)
    m.sets.do("step")

API reference
--------------------------------

.. autoclass:: AgentSet
    :members:
    :inherited-members:
    :autosummary:
    :autosummary-nosignatures:

.. autoclass:: AgentSetRegistry
    :members:
    :inherited-members:
    :autosummary:
    :autosummary-nosignatures:
