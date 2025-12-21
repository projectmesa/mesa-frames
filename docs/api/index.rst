mesa-frames API
===============

.. toctree::
   :caption: Shortcuts
   :maxdepth: 1
   :hidden:

   reference/agents/index
   reference/model
   reference/space/index
   reference/datacollector



Overview
--------

mesa-frames provides a DataFrame-first API for agent-based models. Instead of representing each agent as a distinct Python object, agents are stored in AgentSets (backed by DataFrames) and manipulated via vectorised operations. This leads to much lower memory overhead and faster bulk updates while keeping an object-oriented feel for model structure and lifecycle management.


Mini usage flow
---------------

1. Create a Model and register AgentSets on ``model.sets``.
2. Populate AgentSets with agents (rows) and attributes (columns) via adding a DataFrame to the AgentSet.
3. Implement AgentSet methods that operate on DataFrames
4. Use ``model.sets.do("step")`` from the model loop to advance the simulation; datacollectors and reporters can sample model- and agent-level columns at each step.

.. grid::
    :gutter: 2

    .. grid-item-card:: Manage agent collections
        :link: reference/agents/index
        :link-type: doc

        Create and operate on ``AgentSets`` and ``AgentSetRegisties``: add/remove agents.

    .. grid-item-card:: Model orchestration
        :link: reference/model
        :link-type: doc

        ``Model`` API for registering sets, stepping the simulation, and integrating with datacollectors/reporters.

    .. grid-item-card:: Spatial support
        :link: reference/space/index
        :link-type: doc

        Placement and neighbourhood utilities for ``Grid`` and space

    .. grid-item-card:: Collect simulation data
        :link: reference/datacollector
        :link-type: doc

        Record model- and agent-level metrics over time with ``DataCollector``. Sample columns, run aggregations, and export cleaned frames for analysis.