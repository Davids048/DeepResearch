from react_agent import MultiTurnReactAgent
from evolve.reflector import Reflector
from logger import setup_logging

logger = setup_logging(name=__name__, level=5)

class Evolver:
    """A controller to manage the interaction between task agent, reflector, curators, memory (and other components)."""

    def __init__(
        self, 
        task_agent: MultiTurnReactAgent,
        model_name: str,
        reflector: Reflector
    ):
        self.task_agent = task_agent
        self.model_name = model_name
        self.reflector = reflector

    def evolve(
        self, 
        task:dict,
    ):
        # task agent generates a trajectory 
        trajectory = self.task_agent._run(
            data=task,
            model=self.model_name,
        )
        logger.debug(f"Task agent finished....") 
        logger.debug(f"Trajectory: {trajectory}.\n\n")
        # reflector reflects on the trajectory
        reflection = self.reflector.reflect(trajectory=trajectory)
        logger.debug(f"Reflection: {reflection}.")
    
        # *** curator curates the reflection

        return {
            "trajectory": trajectory,
            "reflection": reflection,
        }


    
