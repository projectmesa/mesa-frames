from copy import deepcopy
from time import time
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd

from mesa_frames.agent import AgentDF


class ModelDF:
    def __new__(cls, *args, **kwargs):
        """Create a new model object and instantiate its RNG automatically
        (adds supports to numpy with respect to base model)."""
        obj = object.__new__(cls)
        obj._seed = kwargs.get("seed")
        if obj._seed is None:
            # We explicitly specify the seed here so that we know its value in
            # advance.
            obj._seed = np.random.SeedSequence().entropy
        # Use default_rng to get a new Generator instance
        obj.random = np.random.default_rng(obj._seed)
        return obj

    def __init__(self, unique_id: int | None = None, space=None):
        """Create a new model. Overload this method with the actual code to
        start the model. Always start with super().__init__() to initialize the
        model object properly.

        Parameters
        ----------
        unique_id : int | None
            The unique_id of the model.
            If None, a random unique_id is assigned using a 64-bit random integer.
        space
            The space where the agents will be placed. Can be None if model does not have a space.

        Attributes
        ----------
        unique_id : int
            The unique_id of the model.
        running : bool
            Indicates if the model is running or not.
        agents : pd.DataFrame | gpd.GeoDataFrame | None
            The dataframe containing the agents of the model.
        agent_types : list[tuple[type[AgentDF], float]] | None
            The list of agent types and their proportions.
        p_agents : dict[type[AgentDF], float] | None
            The dictionary of agents to create. The keys are the types of agents,
            the values are the percentages of each agent type. The sum of the values should be 1.
        space
            The space where the agents will be placed. Can be None if model does not have a space.
        """
        # Initialize default attributes
        self.running: bool = True
        self.agents: pd.DataFrame | gpd.GeoDataFrame | None = None
        self.agent_types: list[tuple[type[AgentDF], float]] | None = None
        self.p_agents: dict[type[AgentDF], float] | None = None
        # self.schedule : BaseScheduler = None

        # Initialize optional parameters
        if not unique_id:
            self.unique_id = np.random.randint(
                low=-9223372036854775808, high=9223372036854775807, dtype="int64"
            )
        else:
            self.unique_id = unique_id
        self.space = space

        # Initialize data collection
        # self.initialize_data_collector(data_collection)

    def get_agents_of_type(self, agent_type: type[AgentDF]) -> pd.Series:
        """Returns a boolean mask of the agents dataframe of the model which corresponds to the agent_type.

        Parameters
        ----------
        agent_type : type[AgentDF]
            The agent_type to get the mask for.
        """
        if self.agents is None:
            raise RuntimeError(
                "You must create agents before getting their masks. Use create_agents() method."
            )
        return self.agents["type"].str.contains(agent_type.__name__)  # type: ignore

    def run_model(self, n_steps: int | None = None, merged_mro: bool = False) -> None:
        """If n_steps are specified, executes model.step() until n_steps are reached.
        Otherwise, until self.running is false (as the default mesa.Model.run_model).

        Parameters
        ----------
        n_steps : int | None
            The number of steps which the model will execute.
            Can be None if a running condition turning false is used.
        merged_mro: bool
            If False, the model will execute one step for each class in p_agent. This is the default behaviour.
            If True, the model will execute one step for each inherited agent type in the order of a "merged" MRO.
            This may increase performance if there are multiple and complex inheritance as each agent_type (even if parents of different classes),
            will be executed only once. Not a viable option if the behavior of a class depends on another.
        """
        if n_steps:
            if not (isinstance(n_steps, int) and n_steps > 0):
                raise TypeError(
                    "n_steps should be an integer greater than 0 or None if a running condition is used"
                )
            for _ in range(n_steps):
                self.step(merged_mro)
        else:
            while self.running:
                self.step(merged_mro)

    def step(self, merged_mro: bool = False) -> None:
        """Executes one step of the model.

        Parameters
        ----------
        merged_mro: bool
            If False, the model will execute one step for each class in p_agent. This is the default behaviour.
            If True, the model will execute one step for each inherited agent type in the order of a "merged" MRO.
            This may increase performance if there are multiple and complex inheritance as each agent_type (even if parents of different classes),
            will be executed only once. Not a viable option if the behavior of a class depends on another.
        """
        if self.agent_types is None or self.p_agents is None:
            raise RuntimeError(
                "You must create agents before running the model. Use create_agents() method."
            )
        if merged_mro:
            for agent_type in self.agent_types:
                agent_type[0].step()
        else:
            for agent in self.p_agents:
                agent.step()

    def reset_randomizer(self, seed: int | None = None) -> None:
        """Reset the model random number generator.

        Parameters:
        ----------
            seed: A new seed for the RNG; if None, reset using the current seed
        """
        if seed is None:
            seed = self._seed
        self.random = np.random.default_rng(seed)
        self._seed = seed

    def create_agents(
        self, n_agents: int, p_agents: dict[type[AgentDF], float]
    ) -> None:
        """Populate the self.agents dataframe.

        Parameters
        ----------
        n_agents : int | None
            The number of agents which the model will create.
        p_agents : dict[type[AgentDF], float]
            The dictionary of agents to create. The keys are the types of agents,
            the values are the percentages of each agent type. The sum of the values should be 1.
        """

        # Verify parameters
        if not (isinstance(n_agents, int) and n_agents > 0):
            raise TypeError("n_agents should be an integer greater than 0")
        if sum(p_agents.values()) != 1:
            raise ValueError("Sum of proportions of agents should be 1")
        if any(p < 0 or p > 1 for p in p_agents.values()):
            raise ValueError("Proportions of agents should be between 0 and 1")

        self.p_agents = p_agents

        start_time = time()
        print("Creating agents: ...")

        mros = [[agent.__mro__[:-1], p] for agent, p in p_agents.items()]
        mros_copy = deepcopy(mros)
        agent_types = []

        # Create a "merged MRO" (inspired by C3 linearization algorithm)
        while True:
            candunique_idate_added = False
            # if all mros are empty, the merged mro is done
            if not any(mro[0] for mro in mros):
                break
            for mro in mros:
                # If mro is empty, continue
                if not mro[0]:
                    continue
                # candunique_idate = head
                candunique_idate = mro[0][0]
                # If candunique_idate appears in the tail of another MRO, skip it for now (because other agent_types depend on it, will be added later)
                if any(
                    candunique_idate in other_mro[0][1:]
                    for other_mro in mros
                    if other_mro is not mro
                ):
                    continue
                else:
                    p = 0
                    for i, other_mro in enumerate(mros):
                        if other_mro[0][0] == candunique_idate:
                            p += other_mro[1]
                            mros[i][0] = other_mro[0][1:]
                        else:
                            continue
                    agent_types.append((candunique_idate, p))  # Safe to add it
                    candunique_idate_added = True
            # If there wasn't any good head, there is an inconsistent hierarchy
            if not candunique_idate_added:
                raise ValueError("Inconsistent hierarchy")
        self.agent_types = list(agent_types)

        # Create a single DF using vars and values for every class
        columns: set[str] = set()
        dtypes: dict[str, str] = {}
        for agent_type in self.agent_types:
            for key, val in agent_type[0].dtypes.items():
                if key not in columns:
                    columns.add(key)
                    dtypes[key] = val

        if "geometry" in columns:
            if not (self.space and hasattr(self.space, "crs")):
                raise ValueError(
                    "You must specify a space with a crs attribute if you want to create GeoAgents"
                )
            self.agents = gpd.GeoDataFrame(
                index=pd.RangeIndex(0, n_agents),
                columns=list(columns),
                crs=self.space.crs,
            )
        else:
            self.agents = pd.DataFrame(
                index=pd.RangeIndex(0, n_agents), columns=list(columns)
            )

        # Populate agents type
        start_index = 0
        for i, (_, p) in enumerate(p_agents.items()):
            self.agents.loc[
                start_index : start_index + int(n_agents * p) - 1, "type"
            ] = str(mros_copy[i][0])
            start_index += int(n_agents * p)

        # Initialize agents
        AgentDF.model = self
        self.update_agents_masks()
        for agent in p_agents:
            agent.__init__()

        # Set dtypes
        for col, dtype in dtypes.items():
            if "int" in dtype and self.agents[col].isna().sum() > 0:  # type: ignore
                warn(
                    f"Pandas does not support NaN values for int{dtype[-2:]} dtypes. Changing dtype to float{dtype[-2:]} for {col}",
                    RuntimeWarning,
                )
                dtypes[col] = "float" + dtype[-2:]
        self.agents = self.agents.astype(dtypes)

        # Set agents' unique_id as index (Have to reassign masks because index changed)
        self.agents.set_index("id", inplace=True)
        self.update_agents_masks()

        print("Created agents: " + "--- %s seconds ---" % (time() - start_time))

    # TODO: implement different data collection frequencies (xw, xd, xh, weekly, daily, hourly, per every step):
    """def initialize_data_collector(
        self,
        model_reporters=None,
        agent_reporters=None,
        tables=None,
    ) -> None:
        if not hasattr(self, "schedule") or self.schedule is None:
            raise RuntimeError(
                "You must initialize the scheduler (self.schedule) before initializing the data collector."
            )
        if self.schedule.get_agent_count() == 0:
            raise RuntimeError(
                "You must add agents to the scheduler before initializing the data collector."
            )
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            tables=tables,
        )
        # Collect data for the first time during initialization.
        self.datacollector.collect(self)"""

    def _initialize_data_collection(self, how="2d") -> None:
        """Initializes the data collection of the model.

        Parameters
        ----------
        how : str
            The frequency of the data collection. It can be 'xd', 'xd', 'xh', 'weekly', 'daily', 'hourly'.
        """
        # TODO: finish implementation of different data collections
        if how == "2d":
            return

    def update_agents_masks(self) -> None:
        """Updates the masks attributes of each agent in self.agent_types.
        Useful after agents are created/deleted or index changes.
        """
        if self.agent_types is None:
            raise RuntimeError(
                "You must create agents before updating their masks. Use create_agents() method."
            )
        for agent_type in self.agent_types:
            agent_type[0].mask = self.get_agents_of_type(agent_type[0])
