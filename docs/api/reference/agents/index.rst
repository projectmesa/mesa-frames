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
---------------------------------

.. tab-set::

   .. tab-item:: AgentSet

      .. tab-set::

         .. tab-item:: Overview

            .. rubric:: Lifecycle / Core

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSet.__init__
               AgentSet.step
               AgentSet.rename
               AgentSet.copy

            .. rubric:: Accessors & Views

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSet.df
               AgentSet.model
               AgentSet.random
               AgentSet.space
               AgentSet.active_agents
               AgentSet.inactive_agents
               AgentSet.index
               AgentSet.pos
               AgentSet.name
               AgentSet.get
               AgentSet.contains
               AgentSet.__len__
               AgentSet.__iter__
               AgentSet.__getitem__
               AgentSet.__contains__

            .. rubric:: Mutators

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSet.add
               AgentSet.remove
               AgentSet.discard
               AgentSet.set
               AgentSet.select
               AgentSet.shuffle
               AgentSet.sort
               AgentSet.do

            .. rubric:: Operators / Internal helpers

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSet.__add__
               AgentSet.__iadd__
               AgentSet.__sub__
               AgentSet.__isub__
               AgentSet.__repr__
               AgentSet.__reversed__

         .. tab-item:: Full API

            .. autoclass:: AgentSet
                :autosummary:
                :autosummary-nosignatures:

   .. tab-item:: AgentSetRegistry

      .. tab-set::

         .. tab-item:: Overview

            .. rubric:: Lifecycle / Core

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSetRegistry.__init__
               AgentSetRegistry.copy
               AgentSetRegistry.rename

            .. rubric:: Accessors & Queries

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSetRegistry.get
               AgentSetRegistry.contains
               AgentSetRegistry.ids
               AgentSetRegistry.keys
               AgentSetRegistry.items
               AgentSetRegistry.values
               AgentSetRegistry.model
               AgentSetRegistry.random
               AgentSetRegistry.space
               AgentSetRegistry.__len__
               AgentSetRegistry.__iter__
               AgentSetRegistry.__getitem__
               AgentSetRegistry.__contains__

            .. rubric:: Mutators / Coordination

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSetRegistry.add
               AgentSetRegistry.remove
               AgentSetRegistry.discard
               AgentSetRegistry.replace
               AgentSetRegistry.shuffle
               AgentSetRegistry.sort
               AgentSetRegistry.do
               AgentSetRegistry.__setitem__
               AgentSetRegistry.__add__
               AgentSetRegistry.__iadd__
               AgentSetRegistry.__sub__
               AgentSetRegistry.__isub__

            .. rubric:: Representation

            .. autosummary::
               :nosignatures:
               :toctree:

               AgentSetRegistry.__repr__
               AgentSetRegistry.__str__
               AgentSetRegistry.__reversed__

         .. tab-item:: Full API

            .. autoclass:: AgentSetRegistry
                :autosummary:
                :autosummary-nosignatures:
