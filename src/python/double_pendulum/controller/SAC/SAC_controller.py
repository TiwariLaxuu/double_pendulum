from double_pendulum.controller.abstract_controller import AbstractController
from stable_baselines3 import SAC

class SACController(AbstractController):
    def __init__(self, model_path, dynamics_func, dt,scaling = True):
        super().__init__()

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt
        self.model.predict([0, 0, 0, 0])
        self.scaling = scaling

    def get_control_output_(self, x, t=None):
        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            action, _state = self.model.predict(obs, deterministic=True)
            print("action", action) 
            u = self.dynamics_func.unscale_action(action)
        else:
            action, _state = self.model.predict(x, deterministic=True)
            print("action", action)
            u = self.dynamics_func.unscale_action(action)
        return u
