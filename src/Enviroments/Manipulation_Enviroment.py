import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.enviroments.manipulation.nut_assembly import NutAssemblySquare
from robosuite.models.task import ManipulationTask
import numpy as np



class Manipulation_Enviroment(NutAssemblySquare):
    """
    Enviroment expanding on Nut Assembly Enviorment provided by RoboSuite. 
    Incorporates a PD controller alongside Gaussian Noise in order to cre-
    -ate 3 different data sets with varying noise. Use case of enviroment
    is to evaluate how Diffusion Policy can correct some amount of noise
    in training data, and to what extent is the present noise too much

    """

    MAX_REWARD = 1.0 + 0.9

    def __init__(self, reward_shaping = True, **kwargs):
        super().__init__(reward_shaping = reward_shaping, **kwargs)

 
    def staged_rewards(self):
        """
        Reward function for task, modified from original NutAssembly Enviroment.
        The original NutAssembly Enivroment gives reward based on if the nut is 
        placed on the current peg (a flat score of 1.0), and has staged rewards
        which give a score between 0 to some value between 0 and 1. The final 
        reward is then normalized to 1[the flat score if placed on peg]
        

        Changes to the reward would include an insert stage and seat stage for staged reward. 
        We can monitor the insert stage by checking if the nut is within an x/y radius of the peg, and is 
        proportional with how down it is on the peg, also considers the amount of force on the peg as well
        """