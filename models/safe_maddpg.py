import torch as th
import torch.nn as nn
import numpy as np
from models.maddpg import MADDPG

from qpsolvers import solve_qp
from utilities.define import * 

class SafeMADDPG(MADDPG):

    def __init__(self, args, target_net=None, constraint_model=None):
        super(SafeMADDPG, self).__init__(args, target_net)
        self.constraint_model = constraint_model
    
    def correct_actions_hard(self, state, actions, k, lb, ub):
        '''
        q = k * a
        '''
        q = th.tensor(k).to(th.float32).to(actions.device) * actions.detach()
        with th.no_grad():
            weight, bias = self.constraint_model.get_coff(state, q)
        weight = weight * k
        actions_numpy = actions.detach().squeeze().cpu().numpy()
        action_dim = actions_numpy.shape[0]
        P = np.eye(action_dim)
        q = -actions_numpy
        lb = np.ones(action_dim) * lb
        ub = np.ones(action_dim) * ub
        G = np.concatenate((-weight, weight), axis=0)
        h = np.concatenate((bias - 0.95, 1.05 - bias), axis=0)
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), G.astype(np.float64), h.astype(np.float64), None, None, lb.astype(np.float64), ub.astype(np.float64))
            assert x is not None
        except:
            return actions_numpy, INFEASIBLE

        if np.linalg.norm(actions_numpy - x) > 1e-3:
            return x, INTERVENTIONS

        return x, UNCHANGED

    def correct_actions_soft(self, state, actions, k, lb, ub):
        '''
        q = k * a
        '''
        q = th.tensor(k).to(th.float32).to(actions.device) * actions.detach()
        with th.no_grad():
            weight, bias = self.constraint_model.get_coff(state, q)
        weight = weight * k
        actions_numpy = actions.detach().squeeze().cpu().numpy()
        action_dim = actions_numpy.shape[0]
        P = np.eye(action_dim)
        q = -actions_numpy
        lb = np.ones(action_dim) * lb
        ub = np.ones(action_dim) * ub
        G = np.concatenate((-weight, weight), axis=0)
        h = np.concatenate((bias - 0.95, 1.05 - bias), axis=0)
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), G.astype(np.float64), h.astype(np.float64), None, None, lb.astype(np.float64), ub.astype(np.float64))
            assert x is not None
        except:
            return actions, INFEASIBLE

        if np.linalg.norm(actions_numpy - x) > 1e-3:
            return x, INTERVENTIONS

        return x, UNCHANGED
