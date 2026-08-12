from robosuite.environments.manipulation.nut_assembly import NutAssemblySquare
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

    MAX_REWARD = 1.0 # Mutal Exclusion between Sparse + Staged Rewards has to make this 1
    HOVER_MULT = 0.7
    INSERT_MULT = 0.8
    SEAT_MULT = 0.9
    INSERT_XY_GAIN, INSERT_Z_GAIN = 30.0, 15.0
    SEAT_XY_GAIN,   SEAT_Z_GAIN   = 18.0, 20.0

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

        r_reach, r_grasp, r_lift, r_hover = super().staged_rewards()

        active_nuts = []
        for i, nut in enumerate(self.nuts):
            if(self.objects_on_pegs[i]):
                continue
            active_nuts.append(nut)
        r_insert = 0.0
        if active_nuts:
            alignments = np.zeros(len(active_nuts))
            descents = np.zeros(len(active_nuts))
            r_inserts = np.zeros(len(active_nuts))
            peg_body_ids = [self.peg1_body_id, self.peg2_body_id]
            z_target = self.table_offset[2] + 0.01
            for i, nut in enumerate(active_nuts):
                valid_obj = False
                peg_pos_xy = None
                for nut_name, idn in self.nut_to_id.items():
                    if nut_name in nut.name.lower():
                        peg_pos_xy = np.array(self.sim.data.body_xpos[peg_body_ids[idn]])[:2]
                        valid_obj = True
                        break
                if not valid_obj:
                    raise Exception("Got invalid object to reach: {}".format(nut.name))
                ob_z = self.sim.data.body_xpos[self.obj_body_id[nut.name]][2]
                ob_xy = self.sim.data.body_xpos[self.obj_body_id[nut.name]][:2]
                dist_xy = np.linalg.norm(peg_pos_xy - ob_xy)
                z_dist = np.maximum(-1*(z_target - ob_z), 0)
                alignments[i] = (1 - np.tanh(self.INSERT_XY_GAIN*dist_xy))
                descents[i] = (1 - np.tanh(self.INSERT_Z_GAIN * z_dist))
                r_inserts[i] = r_hover + alignments[i] * descents[i] * (self.INSERT_MULT - self.HOVER_MULT)
            r_insert = np.max(r_inserts)
        r_seat = 0.0
        if active_nuts: 
            r_seats = np.zeros(len(active_nuts))
            peg_body_ids = [self.peg1_body_id, self.peg2_body_id]
            for i, nut in enumerate(active_nuts):
                valid_obj = False
                peg_pos = None
                for nut_name, idn in self.nut_to_id.items():
                    if nut_name in nut.name.lower():
                        peg_pos = np.array(self.sim.data.body_xpos[peg_body_ids[idn]])[:3]
                        valid_obj = True
                        break
                if not valid_obj:
                    raise Exception("Got invalid object to reach: {}".format(nut.name))
                ob_pos = self.sim.data.body_xpos[self.obj_body_id[nut.name]][:3]
                dist_xy = np.linalg.norm(peg_pos[:2] - ob_pos[:2])
                z_dist = np.maximum(-1*(z_target - ob_pos[2]), 0)
                gripper_dist = min([np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - ob_pos) for arm in self.robots[0].arms])
                r_seats[i] = r_insert + (1 - np.tanh(self.SEAT_XY_GAIN*dist_xy)) * (1 - np.tanh(self.SEAT_Z_GAIN*z_dist)) * (min(np.tanh(10 * gripper_dist)/0.4, 1)) * (self.SEAT_MULT - self.INSERT_MULT)
            r_seat = np.max(r_seats)
        return r_reach, r_grasp, r_lift, r_hover, r_insert, r_seat
            


