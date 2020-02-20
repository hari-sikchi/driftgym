#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Simulation environment using vehicle model defined in model.py.
"""

import yaml

import numpy as np
import gym
from gym import spaces



from driftgym.envs.model.model import BrushTireModel, LinearTireModel
from driftgym.envs.renderer import _Renderer


class VehicleEnv(gym.Env):
    """
    Simulation environment for a FFAST RC car.
    """

    _MIN_VELOCITY = 0.5
    _MAX_VELOCITY = 10.0
    _MAX_STEER_ANGLE = np.pi / 6
    _HORIZON_LENGTH = 1000


    def __init__(
            self,
            target_velocity=1.0,
            dt=0.035,
            model_type='BrushTireModel',
            robot_type='RCCar',
            mu_s=1.37,
            mu_k=1.96
    ):
        """
        Initialize environment parameters.
        """
        # Load estimated parameters for robot
        if robot_type == 'RCCar':
            stream = open('/Users/harshit/work/driftgym/driftgym/envs/model/model_params/rccar.yml','r')
            self._params = yaml.load(stream, Loader=yaml.FullLoader)
        elif robot_type == 'MRZR':
            stream = open('driftgym/envs/model/model_params/mrzr.yml','r')
            self._params = yaml.load(stream, Loader=yaml.FullLoader)
        else:
            raise ValueError('Unrecognized robot type')

        # Instantiate vehicle model for simulation
        self._state = None
        self._action = None
        self._unsafe_action=None
        self.target_velocity = target_velocity
        if model_type == 'BrushTireModel':
            self._model = BrushTireModel(self._params, mu_s, mu_k)
        elif model_type == 'LinearTireModel':
            self._model = LinearTireModel(self._params, mu_s, mu_k)
        else:
            raise ValueError('Invalid vehicle model type')

        # Time between each simulation iteration
        # Note: dt is measured to be 0.035, but we train with longer dt
        #       for more stability in commanded actions.
        self._dt = dt
        self.steps_taken = 0
        # Instantiates object handling simulation renderings
        self._renderer = None


    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(6,))


    @property
    def action_space(self):
        low = np.array([VehicleEnv._MIN_VELOCITY,
            -VehicleEnv._MAX_STEER_ANGLE])
        high = np.array([VehicleEnv._MAX_VELOCITY,
            VehicleEnv._MAX_STEER_ANGLE])
        return spaces.Box(low=low, high=high)


    @property
    def horizon(self):
        return VehicleEnv._HORIZON_LENGTH


    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = None
        self._state = self.get_initial_state
        observation = self.state_to_observation(self._state)
        self.steps_taken=0
        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation


    def step(self, action,unsafe_action=None):
        """
        Move one iteration forward in simulation.
        """
        # Place limits on action based on mechanical constraints
        self.steps_taken+=1
        action_min = [VehicleEnv._MIN_VELOCITY, -VehicleEnv._MAX_STEER_ANGLE]
        action_max = [VehicleEnv._MAX_VELOCITY, VehicleEnv._MAX_STEER_ANGLE]
        action = np.clip(action, a_min=action_min, a_max=action_max)

        self._action = action
        if unsafe_action is not None:
            self._unsafe_action=unsafe_action
        nextstate = self._model.state_transition(self._state, action,
                self._dt)
        self._state = nextstate
        reward, info = self.get_reward(nextstate, action)
        observation = self.state_to_observation(nextstate)
        if(self.steps_taken>=self.horizon):
            done=True
        else:
            done=False

        return observation,reward,done,info


    def get_linear_dynamics(self,prev_action=None):
        current_observation = self.state_to_observation(self._state)
        linear_dynamics_a = np.zeros((self.action_space.shape[0],self.observation_space.shape[0]))
        for action_dim in range(self.action_space.shape[0]):
            d_action = 0.001
            if prev_action is None:
                action = np.zeros(self.action_space.shape[0])
            else:
                action = prev_action.copy()
            next_base_state = self._model.state_transition(self._state, action,
                    self._dt)
            next_base_observation =  self.state_to_observation(next_base_state)
            action[action_dim]+=d_action
            nextstate = self._model.state_transition(self._state, action,
                    self._dt)


            next_observation = self.state_to_observation(nextstate)
            linear_dynamics_a[action_dim,:]=(next_observation-current_observation)/d_action        
            
        return linear_dynamics_a,next_base_observation



    def render(self):
        """
        Render simulation environment.
        """
        if self._renderer == None:
            self._renderer = _Renderer(self._params,
                    self.__class__.__name__)
        # print(self._state)
        self._renderer.update(self._state, self._action,unsafe_action=self._unsafe_action)


    # def log_diagnostics(self, paths):
    #     """
    #     Log extra information per iteration based on collected paths.
    #     """
    #     dists = []
    #     vels = []
    #     kappas = []
    #     for path in paths:
    #         dists.append(path['env_infos']['dist'])
    #         vels.append(path['env_infos']['vel'])
    #         kappas.append(path['env_infos']['kappa'])
    #     dists = np.abs(dists)
    #     vels = np.abs(vels)
    #     kappas = np.abs(kappas)

    #     logger.record_tabular('AverageAbsDistance', np.mean(dists))
    #     logger.record_tabular('AverageAbsVelocity', np.mean(vels))
    #     logger.record_tabular('MaxAbsDistance', np.max(dists))
    #     logger.record_tabular('MaxAbsVelocity', np.max(vels))
    #     logger.record_tabular('AverageKappa', np.mean(kappas))
    #     logger.record_tabular('MaxKappa', np.max(kappas))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        raise NotImplementedError


    def get_reward(self, state, action):
        """
        Reward function definition. Returns reward, a scalar, and info, a
        dictionary that must contain the keys 'dist' (closest distance to
        trajectory) and 'vel' (current velocity).
        """
        raise NotImplementedError


    def state_to_observation(self, state):
        """
        Prepare state to be read as input to neural network.
        """
        raise NotImplementedError

