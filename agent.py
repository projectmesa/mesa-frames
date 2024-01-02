from dataclasses import dataclass
from uuid import uuid4
import pandas as pd
from numpy.random import randint

@dataclass
class AgentParams():
    """The AgentParams class is a dataclass that contains the parameters of an Agent.
    It is used to initialize the Agent class.
    """
    pass

class Agent():
    """The Agent class is a class that defines the basic attributes and methods of an agent.
    It is used as a parent class for all the other agents.
    
    Parameters:
    ----------
    params : AgentParams
        The parameters of the Agent. Default: None
    dtypes : dict[str, str]
        The data types of the attributes of the Agent. Default: {'id' : 'int64', 'type' : 'str'}
    values : dict[str, any]
        The values of the attributes of the Agent. Default: {'id' : lambda: uuid4().int >> 64, 'status' : 'free'}
    model : Model
        The model of the simulation where the Agent is used. See streetcrime/model/model.py. Default: None
    agents : gpd.GeoDataFrame
        The GeoDataFrame containing all Agents. Default: None
    """
    
    params : AgentParams = None
    dtypes : dict[str, str] = {
        'id' : 'int64',
        'type' : 'str',
        }
    model : 'Model' = None
    mask : pd.Series = None
    
    @classmethod
    def __init__(cls):
        cls.model.agents.loc[cls.mask, 'id'] = randint(low=-9223372036854775808, high=9223372036854775807, size = cls.mask.sum(), dtype='int64')
    
class GeoAgent(Agent):
    dtypes : dict[str, str] = {
        'geometry' : 'geometry'
    }
