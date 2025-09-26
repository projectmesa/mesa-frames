from dataclasses import dataclass
import mesa_frames
import mesa

@dataclass
class FramesSimulationResult:
    """Container for example simulation outputs.

    The dataclass is intentionally permissive: some backends only provide
    `metrics`, while others also return `agent_metrics`.
    """

    datacollector: mesa_frames.DataCollector

@dataclass
class MesaSimulationResult:
    """Container for example simulation outputs.

    The dataclass is intentionally permissive: some backends only provide
    `metrics`, while others also return `agent_metrics`.
    """

    datacollector: mesa.DataCollector