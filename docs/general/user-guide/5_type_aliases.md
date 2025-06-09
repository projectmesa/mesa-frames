# 📌 Type Aliases Documentation

## What are Type Aliases?

A **type alias** assigns a meaningful name to an existing type. It helps improve **code readability** and **maintainability**, especially when dealing with **complex types**.

### **Example:**

```python
GridCapacity = PandasGridCapacity | PolarsGridCapacity
```

Here, `GridCapacity` is a shorthand for both Pandas and Polars grid capacities.

---

## **📂 Type Aliases by File**

### **📁 types_.py** *(Core Type Aliases)*

#### **🔹 Agnostic Types**

- `AgnosticMask` → Flexible type for masks (`Any | Sequence[Any] | None`)
- `AgnosticAgentMask` → Selection filters for agents (`Sequence[int] | int | Literal["all", "active"] | None`)
- `AgnosticIds` → Unique agent IDs (`int | Collection[int]`)

#### **🔹 Pandas Types**

- `PandasMask` → Mask type for Pandas (`pd.Series | pd.DataFrame | AgnosticMask`)
- `AgentPandasMask` → Agent-specific mask for Pandas (`AgnosticAgentMask | pd.Series | pd.DataFrame`)
- `PandasIdsLike` → ID representations in Pandas (`AgnosticIds | pd.Series | pd.Index`)
- `PandasGridCapacity` → Grid capacity stored in a NumPy array (`ndarray`)

#### **🔹 Polars Types**

- `PolarsMask` → Mask type for Polars (`pl.Expr | pl.Series | pl.DataFrame | AgnosticMask`)
- `AgentPolarsMask` → Agent-specific mask for Polars (`AgnosticAgentMask | pl.Expr | pl.Series | pl.DataFrame | Sequence[int]`)
- `PolarsIdsLike` → ID representations in Polars (`AgnosticIds | pl.Series`)
- `PolarsGridCapacity` → Grid capacity using Polars expressions (`list[pl.Expr]`)

#### **🔹 Generic Types**

- `DataFrame` → Supports Pandas & Polars DataFrames (`pd.DataFrame | pl.DataFrame`)
- `DataFrameInput` → Accepted DataFrame inputs (`dict[str, Any] | Sequence[Sequence] | DataFrame`)
- `Series` → Pandas or Polars Series (`pd.Series | pl.Series`)
- `Index` → Indexing type (`pd.Index | pl.Series`)
- `BoolSeries` → Boolean Series representation (`pd.Series | pl.Series`)
- `Mask` → General mask type (`PandasMask | PolarsMask`)
- `AgentMask` → General agent mask (`AgentPandasMask | AgentPolarsMask`)
- `IdsLike` → ID representations (`AgnosticIds | PandasIdsLike | PolarsIdsLike`)
- `ArrayLike` → Array representation (`ndarray | Series | Sequence`)

#### **🔹 Time-Related Types**

- `TimeT` → Time-based values (`float | int`)

#### **🔹 Space Types**

- `NetworkCoordinate` → Network-based coordinate (`int | DataFrame`)
- `GridCoordinate` → Grid-based coordinate (`int | Sequence[int] | DataFrame`)
- `DiscreteCoordinate` → Discrete spatial coordinate (`NetworkCoordinate | GridCoordinate`)
- `ContinousCoordinate` → Continuous spatial coordinate (`float | Sequence[float] | DataFrame`)
- `SpaceCoordinate` → General space coordinate (`DiscreteCoordinate | ContinousCoordinate`)

- `GridCapacity` → Grid storage capacity (`PandasGridCapacity | PolarsGridCapacity`)
- `NetworkCapacity` → Network storage capacity (`DataFrame`)
- `DiscreteSpaceCapacity` → Discrete space capacity alias (`GridCapacity | NetworkCapacity`)

---

### **📁 space.py**

- `ESPG` → Defines an integer alias for EPSG codes in spatial reference systems (`int`)

---

### **📁 agents.py**

#### **🔹 Agent Identifiers & Storage**

- `IdsLike` → `AgnosticIds | PandasIdsLike | PolarsIdsLike`Used for identifying agents in collections
- `AgentMask` → `AgentPandasMask | AgentPolarsMask`Used for filtering/selecting agents

#### **🔹 Agent Data Representation**

- `BoolSeries` → `pd.Series | pl.Series`Boolean representation of agents
- `Series` → `pd.Series | pl.Series`Holds a Pandas or Polars Series for agent data
- `DataFrameInput` → `dict[str, Any] | Sequence[Sequence] | DataFrame`Accepted input formats for DataFrames

---

### **📁 mixin.py**

#### **🔹 Utility Type Aliases**

- `BoolSeries` → `pd.Series | pl.Series` Used for boolean operations in mixins
- `DataFrame` → `pd.DataFrame | pl.DataFrame`Used in mixins for defining DataFrame operations
- `Index` → `pd.Index | pl.Series`Used for indexing operations in mixins
- `Mask` → `PandasMask | PolarsMask`General mask type
- `Series` → `pd.Series | pl.Series` Pandas or Polars series representation

---
