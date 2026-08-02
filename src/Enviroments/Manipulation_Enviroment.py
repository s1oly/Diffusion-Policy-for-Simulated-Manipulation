import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.enviroments.manipulation.nut_assembly import NutAssemblySquare
from robosuite.models.task import ManipulationTask
import numpy as np

#Child inherits all attributes, use self() when wanting to overide methods 

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

    def rewards(self, action = None):
        reward = super().rewards()
        return reward/self.MAX_REWARD

 
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

        r_reach, r_grasp, r_lift, r_hover = super().staged_rewards()
        insert_mult = 0.8
        seat_mult = 0.9 # Look at on_peg determination and then have less rigor to award seat

        active_nuts = []
        for i, nut in enumerate(self.nuts):
            if(self.objects_on_pegs[i]):
                continue
            active_nuts.append(nut)
        r_insert = 0.0
        if active_nuts:
            r_insert = np.zeros(len(active_nuts))
            peg_body_ids = [self.peg1_body_id, self.peg2_body_id]
            



