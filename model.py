from mesa_frames.agent import Agent
from mesa_frames.space import Space, GeoSpace
import pandas as pd
import geopandas as gpd
from uuid import uuid4
from time import time
from copy import deepcopy
from warnings import warn

class Model():

    def __init__(self,
                 space: Space = None,
                 n_steps: int = None,
                 p_agents: dict[type[Agent], float] = None,
                 n_agents: int = None,
                 data_collection: str = '2d', #TODO: implement different data collection frequencies (xw, xd, xh, weekly, daily, hourly, per every step)
                 **kwargs):
        
        #self._verify_parameters(data_collection)
        
        # Initialize model attributes
        self.id = uuid4().int >> 64
        self.n_agents : int = None
        self.agent_types = None
        self.agents : pd.DataFrame | gpd.GeoDataFrame = None
        self.p_agents = p_agents
        self.space = space
        self.n_steps = n_steps
        self.n_agents = n_agents
            
        # Initialize data collection
        self._initialize_data_collection(data_collection)
                    
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def create_agents(self,
                      p_agents : dict[type[Agent], float] = None,
                      n_agents : int = None) -> None:
        """Creates the agents of the model and adds them to the schedule and the space
        
        Parameters
        ----------
        num_agents : int
            The number of agents to create
        agents : dict[type[agent], float]
            The dictionary of agents to create. The keys are the types of agents and the values are the percentages of each agent type to create.
        """
        start_time = time()
        print("Creating agents: ...")

        mros = [[agent.__mro__[:-1], p] for agent, p in p_agents.items()]
        mros_copy = deepcopy(mros)
        agent_types = []
        
        # Create a "merged MRO" (inspired by C3 linearization algorithm)
        while True:
            candidate_added = False
            #if all mros are empty, the merged mro is done
            if not any(mro[0] for mro in mros):
                break
            for mro in mros:
                # If mro is empty, continue
                if not mro[0]:
                    continue
                # candidate = head
                candidate = mro[0][0]
                # If candidate appears in the tail of another MRO, skip it for now (because other agent_types depend on it, will be added later)
                if any(candidate in other_mro[0][1:] for other_mro in mros if other_mro is not mro):
                    continue 
                else:
                    p = 0
                    for i, other_mro in enumerate(mros):
                        if other_mro[0][0] == candidate:
                            p += other_mro[1]
                            mros[i][0] = other_mro[0][1:]
                        else:
                            continue
                    agent_types.append((candidate, p)) #Safe to add it
                    candidate_added = True
            # If there wasn't any good head, there is an inconsistent hierarchy
            if not candidate_added:
                raise ValueError("Inconsistent hierarchy")
        self.agent_types = list(reversed(agent_types))
        
        ### Create DataFrame using vars and values for every class
        columns = set()
        dtypes = {}
        for agent_type in self.agent_types:
            for key, val in agent_type[0].dtypes.items():
                if key not in columns:
                    columns.add(key)
                    dtypes[key] = val
    
        if isinstance(self.space, GeoSpace):
            self.agents = gpd.GeoDataFrame(index=pd.RangeIndex(0, n_agents), columns = list(columns), crs = self.space.crs)
        else:
            self.agents = pd.DataFrame(index=pd.RangeIndex(0, n_agents), columns = list(columns))        
        
        #Populate type column
        start_index = 0
        for i, (_, p) in enumerate(p_agents.items()):
            self.agents.loc[start_index:start_index + int(n_agents * p)-1, 'type'] = str(mros_copy[i][0])
            start_index += int(n_agents * p)
        
        #Initialize model and mask attributes for agents
        Agent.model = self
        self.update_agents_masks()
        
        #Execute init method for every agent
        for agent in p_agents:
            agent.__init__()
        
        for col, dtype in dtypes.items():
            if 'int' in dtype and self.agents[col].isna().sum() > 0:
                #warn(f'Pandas does not support NaN values for int{dtype[-2:]} dtypes. Changing dtype to float{dtype[-2:]} for {col}', RuntimeWarning)
                dtypes[col] = 'float' + dtype[-2:]
        self.agents = self.agents.astype(dtypes)
        self.agents.set_index('id', inplace=True)  
        #Have to reassign masks because index changed
        self.update_agents_masks()
        print("Created agents: " + "--- %s seconds ---" % (time() - start_time))
        
    def run_model(self) -> None:
        for _ in range(self.n_steps):
            self.step()
    
    def step(self) -> None:
        for agent in self.p_agents:
            agent.step()
        
    def _initialize_data_collection(self, how = '2d') -> None:
        """Initializes the data collection of the model.
        
        Parameters
        ----------
        how : str
            The frequency of the data collection. It can be 'xd', 'xd', 'xh', 'weekly', 'daily', 'hourly'.
        """
        #TODO: finish implementation of different data collections
        if how == '2d':
            return
     
    def update_agents_masks(self) -> None:
        for agent_type in self.agent_types:
            agent_type[0].mask = self.agents['type'].str.contains(str(agent_type[0]))
               
    '''def _verify_parameters(self, p_agents, n_agents, data_collection) -> None:
        """Verifies that the parameters of the model are correct.
        """
        if sum(p_agents.values()) != 1:
            raise ValueError("Sum of proportions of agents should be 1")
        if any(p < 0 or p > 1 for p in p_agents.values()):
            raise ValueError("Proportions of agents should be between 0 and 1")
        
        if n_agents <= 0 or n_agents % 1 != 0:
            raise ValueError("Number of agents should be an integer greater than 0")
        
        #TODO: implement data_collection verification'''