import gymnasium as gym
from gymnasium import spaces

import bumps.names as bumps
from bumps import fitters
from bumps.curve import Curve

import libpymcr as lb
#m = lb.Matlab('C:/Users/jyw1/Desktop/SpinW/spinw/PySpinW.ctf')

import numpy as np
import random
import math

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class SpinEnv(gym.Env):
    envID = 0
    # metadata = {'render.modes': ['human']}

    def __init__(self, logging=True):
        super(SpinEnv, self).__init__()
        #self.id = self.envID
        self.id = random.randint(0, 10000)
        #print("Creating ID = ", self.id)
        SpinEnv.envID += 1

        self.logging=logging

        self.m = lb.Matlab('C:/Users/jyw1/Desktop/SpinW/spinw/PySpinW.ctf')
        self.point_density = 50
        self.q_arr = np.linspace(start=0, stop=1, num=self.point_density)

        # Define action space based on possible combinations of scans
        self.action_options = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.action_space = spaces.Discrete(len(self.action_options))

        # Set up real model - currently next-nearest neighbor
        self.real_j1 = -5
        self.real_j2 = -1
        j1_mat = make_J(self.real_j1)
        j2_mat = make_J(self.real_j2)
        self.real_model = self.m.spinw()
        self.real_model.genlattice('lat_const', [3, 3, 18], 'angled', [90, 90, 90])
        self.real_model.addatom(r=[0, 0, 0], S=[1], label='MCu1')
        self.real_model.gencoupling('maxDistance', float(7))
        self.real_model.addmatrix('value', j1_mat, 'label', 'J1')
        self.real_model.addcoupling('mat', 'J1', 'bond', float(1))
        self.real_model.addmatrix('value', j2_mat, 'label', 'J2')
        self.real_model.addcoupling('mat', 'J2', 'bond', float(2))
        self.real_model.genmagstr('mode', 'direct', 'k', [0, 0, 0], 'n', [1, 0, 0], 'S', [[0], [1], [0]])

        # Starting values
        self.start_n_j1 = -3
        self.start_nn_j1 = -3
        self.start_nn_j2 = -3
        n_j1_mat = make_J(self.start_n_j1)
        nn_j1_mat = make_J(self.start_nn_j1)
        nn_j2_mat = make_J(self.start_nn_j2)

        # Nearest neighbor test model
        self.n_test_model = self.m.spinw()
        self.n_test_model.genlattice('lat_const', [3, 3, 18], 'angled', [90, 90, 90])
        self.n_test_model.addatom(r=[0, 0, 0], S=[1], label='MCu1')
        self.n_test_model.gencoupling('maxDistance', float(7))
        self.n_test_model.addmatrix('value', n_j1_mat, 'label', 'J1')
        self.n_test_model.addcoupling('mat', 'J1', 'bond', float(1))
        self.n_test_model.genmagstr('mode', 'direct', 'k', [0, 0, 0], 'n', [1, 0, 0], 'S', [[0], [1], [0]])

        # Next-nearest neighbor test model
        self.nn_test_model = self.m.spinw()
        self.nn_test_model.genlattice('lat_const', [3, 3, 18], 'angled', [90, 90, 90])
        self.nn_test_model.addatom(r=[0, 0, 0], S=[1], label='MCu1')
        self.nn_test_model.gencoupling('maxDistance', float(7))
        self.nn_test_model.addmatrix('value', nn_j1_mat, 'label', 'J1')
        self.nn_test_model.addcoupling('mat', 'J1', 'bond', float(1))
        self.nn_test_model.addmatrix('value', nn_j2_mat, 'label', 'J2')
        self.nn_test_model.addcoupling('mat', 'J2', 'bond', float(2))
        self.nn_test_model.genmagstr('mode', 'direct', 'k', [0, 0, 0], 'n', [1, 0, 0], 'S', [[0], [1], [0]])

        self.reward_scale = 100
        self.steps = 0
        self.episodeNum = 0
        self.info = {}

        self.q_start = [0, 0, 0]
        self.q_end = [0, 0, 0]

        # Logging
        self.chisqds = []
        self.j1s = []
        self.j2s = []
        self.actions = []
        self.bic_diffs = []
        self.n_bics = []
        self.nn_bics = []
        self.n_nllfs = []
        self.nn_nllfs = []
        self.rewards = []
        self.end_j1s = []
        self.convergs_n = []
        self.convergs_nn = []
        self.correct_ends = []
        self.bads = []
        self.good_n = False
        self.good_nn = False
        self.totReward = 0
        self.chose_bad = False
        self.bad_step = -1

    def step(self, action):
        print("entering step in environment", self.id)
        self.steps += 1
        self.q_end = self.action_options[action]
        if self.q_end == [0, 0, 1]:
            self.chose_bad = True
            self.bad_step = self.steps

        # "scan"
        data = self.real_model.spinwave((self.q_start, self.q_end, self.point_density))["omega"]
        #print(data.shape)
        #data = data * (1+0.05*np.random.randn(*data.shape))
        #print("spinwave")

        reward = -self.reward_scale
        print("* creating curve")
        # nearest neighbor model
        near_M = Curve(near_calcSpec, self.q_arr, data, j1=self.start_n_j1, q_start=self.q_start, q_end=self.q_end,
                       point_density=self.point_density, model=self.n_test_model)
        near_M.j1.range(-10, -0.1)
        print("* doing update")
        near_M.update()
        print("* done update")
        # next nearest neighbor model
        next_near_M = Curve(next_near_calcSpec, self.q_arr, data, j1=self.start_nn_j1, j2=self.start_nn_j2,
                            q_start=self.q_start, q_end=self.q_end, point_density=self.point_density,
                            model=self.nn_test_model)
        next_near_M.j1.range(-10, -0.1)
        next_near_M.j2.range(-10, -0.1)
        next_near_M.update()
        # fit
        near_js, near_dx, near_chisq, near_params = self.fit(near_M)
        self.start_n_j1 = near_js[0]
        next_near_js, next_near_dx, next_near_chisq, next_near_params = self.fit(next_near_M)
        self.start_nn_j1 = next_near_js[0]
        self.start_nn_j2 = next_near_js[1]
        # nllf
        n_m = Curve(near_calcSpec, self.q_arr, data, j1=self.start_n_j1, q_start=self.q_start, q_end=self.q_end,
                    point_density=self.point_density, model=self.n_test_model)
        nn_m = Curve(next_near_calcSpec, self.q_arr, data, j1=self.start_nn_j1, j2=self.start_nn_j2,
                     q_start=self.q_start, q_end=self.q_end, point_density=self.point_density, model=self.nn_test_model)
        #print("***** about to define fit problem")
        near_problem = bumps.FitProblem(n_m)
        next_near_problem = bumps.FitProblem(nn_m)
        #print("***** evalulating nllf")
        near_nllf = near_problem.model_nllf()
        next_near_nllf = next_near_problem.model_nllf()
        #print("***** finished nllf")
        # BIC
        N = self.point_density  # ??????
        near_BIC = 2 * near_nllf + math.log(N) * 1
        next_near_BIC = 2 * next_near_nllf + math.log(N) * 2
        bic_diff = abs(near_BIC - next_near_BIC)

        state = np.array([action])

        # Reward function
        if bic_diff > 10:
            reward += 150
        else:
            reward += 10 * (bic_diff)
        chisq = near_chisq if near_BIC < next_near_BIC else next_near_chisq
        dx = near_dx[0] if near_BIC < next_near_BIC else next_near_dx[0]
        self.correct = False if near_BIC < next_near_BIC else True
        if chisq < 1 and dx < 100:
            reward += 50

        self.totReward += reward

        # Add things to log
        self.j1s.append(self.start_nn_j1)
        self.j2s.append(self.start_nn_j2)
        self.actions.append(action)
        self.chisqds.append(chisq)
        self.bic_diffs.append(bic_diff)
        self.n_bics.append(near_BIC)
        self.nn_bics.append(next_near_BIC)
        self.n_nllfs.append(near_nllf)
        self.nn_nllfs.append(next_near_nllf)
        if not self.good_n and abs(self.start_n_j1 - self.real_j1) < 0.1:
            self.convergs_n.append(self.steps)
            self.good_n = True
        if not self.good_nn and abs(self.start_nn_j1 - self.real_j1) < 0.1 and abs(
                self.start_nn_j2 - self.real_j2) < 0.1:
            self.convergs_nn.append(self.steps)
            self.good_nn = True

        # Terminal conditions
        terminal = False
        if chisq < 0.05 and dx < 100:  # less than or equal to?
            print("terminated: excellent conditions")
            terminal = True
            self.log()
        elif (self.steps >= len(self.action_options)):
            print("terminated: too long")
            terminal = True
            self.log()

        # Truncated conditions
        truncated = False

        print("step output", state, reward, terminal, truncated, self.info)
        return state, reward, terminal, truncated, self.info

    def reset(self, seed=None):
        # seed isn't used
        self.start_n_j1 = -3

        self.start_nn_j1 = -3
        self.start_nn_j2 = -1

        self.steps = 0

        self.chisqds = []
        self.bic_diffs = []
        self.n_bics = []
        self.nn_bics = []
        self.n_nllfs = []
        self.nn_nllfs = []
        self.j1s = []
        self.j2s = []
        self.actions = []
        self.good_n = False
        self.good_nn = False
        self.totReward = 0
        self.chose_bad = False
        self.bad_step = -1

        return np.array([0]), self.info

    def render(self, mode='human'):
        # useless
        print("rendering!")

    def close(self):
        # useless
        pass

    def fit(self, model):
        print("% starting fit")
        model.update()
        problem = bumps.FitProblem(model)
        result = fitters.fit(problem, method='lm')
        print("% done fitting")
        for p, v in zip(problem._parameters, result.dx):
            p.dx = v
        return result.x, result.dx, problem.chisq(), problem._parameters

    # logs omitted for now
    def log(self):
        if not self.logging:
            return

        import os

        print("$$$$ logging from ", os.getpid())
        pid = os.getpid()

        directory = "C:/Users/jyw1/Desktop/SpinW/spinw/logs/"

        directory += f"experiment{EXPERIMENT}/{pid}/"

        for p in "chis bics n_bics nn_bics n_nllfs nn_nllfs j1s j2s actions".split():
            os.makedirs(directory + p, exist_ok=True)

        self.episodeNum += 1

        tag = f"{self.episodeNum:05d}"

        filename = directory + "chis/chiLog-" + tag + ".npy"
        np.savetxt(filename, self.chisqds)

        filename = directory + "bics/bicLog-" + tag + ".npy"
        np.savetxt(filename, self.bic_diffs)

        filename = directory + "n_bics/bicLog-" + tag + ".npy"
        np.savetxt(filename, self.n_bics)

        filename = directory + "nn_bics/bicLog-" + tag + ".npy"
        np.savetxt(filename, self.nn_bics)

        filename = directory + "n_nllfs/nllfLog-" + tag + ".npy"
        np.savetxt(filename, self.n_nllfs)

        filename = directory + "nn_nllfs/nllfLog-" + tag + ".npy"
        np.savetxt(filename, self.nn_nllfs)

        filename = directory + "j1s/j1Log-" + tag + ".npy"
        np.savetxt(filename, self.j1s)

        filename = directory + "j2s/j2Log-" + tag + ".npy"
        np.savetxt(filename, self.j2s)

        filename = directory + "actions/actionLog-" + tag + ".npy"
        np.savetxt(filename, self.actions)

        filename = directory + "convergs_n.npy"
        np.savetxt(filename, self.convergs_n)

        filename = directory + "convergs_nn.npy"
        np.savetxt(filename, self.convergs_nn)

        if self.correct:
            self.correct_ends.append(1)
        else:
            self.correct_ends.append(0)

        filename = directory + "correct_ends.npy"
        np.savetxt(filename, self.correct_ends)

        self.bads.append(self.bad_step)
        filename = directory + "bads.npy"
        np.savetxt(filename, self.bads)

        self.end_j1s.append(self.start_n_j1)
        filename = directory + "end_j1s.npy"
        np.savetxt(filename, self.end_j1s)

        self.rewards.append(self.totReward)
        filename = directory + "rewards.npy"
        np.savetxt(filename, self.rewards)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=len(self.action_options), shape=(self.steps + 1,), dtype=np.int32)


def make_J(j):
    return j * np.identity(3)


def next_near_setJ(model, j1, j2):
    j1_mat = make_J(j1)
    j2_mat = make_J(j2)
    model.addmatrix('value', j1_mat, 'label', 'J1')
    model.addcoupling('mat', 'J1', 'bond', float(1))
    model.addmatrix('value', j2_mat, 'label', 'J2')
    model.addcoupling('mat', 'J2', 'bond', float(2))


def next_near_calcSpec(q_arr, j1, j2, q_start=None, q_end=None, point_density=None, model=None):
    next_near_setJ(model, j1, j2)
    omega = model.spinwave((q_start, q_end, point_density))["omega"]
    return omega


def near_setJ(model, j1):
    mat = make_J(j1)
    model.addmatrix('value', mat, 'label', 'J1')
    model.addcoupling('mat', 'J1', 'bond', float(1))


def near_calcSpec(q_arr, j1, q_start=None, q_end=None, point_density=None, model=None):
    # print("q_start:", q_start)
    near_setJ(model, j1)
    omega = model.spinwave((q_start, q_end, point_density))["omega"]
    return omega


# Instantiate the env
#env = SpinEnv()

def start(reload):
    log_dir = f'C:/Users/jyw1/Desktop/SpinW/spinw/logs/Trained/experiment{EXPERIMENT}'
    trained_dir = f'C:/Users/jyw1/Desktop/SpinW/spinw/logs/Trained/experiment{EXPERIMENT}'
    n_epochs = 8 # 10 hours -> 10 min per 2 epochs -> 2 epochs
    n_envs = 1
    n_steps = 2048
    batch_size = 64
    total_timesteps = 15000
    env = make_vec_env(lambda: SpinEnv(logging=not reload), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # check environment
    # check_env(env)
    # Define and Train the agent
    print("reload = ", reload)
    #print("PPO run", env, type(env))
    if reload:
        model = PPO.load(trained_dir, env=env)
        print("model loaded as ", model)
        mean_reward = evaluate_policy(model, n_eval_episodes=n_envs, env=env)
        print("MEAN REWARD FROM SAVED MODEL", mean_reward)
    else:
        model = PPO('MlpPolicy', env, verbose = 2, n_epochs=n_epochs, n_steps=n_steps, batch_size=batch_size)
        trained = model.learn(total_timesteps=total_timesteps, progress_bar=True)
        trained.save(log_dir)

EXPERIMENT = "longC"

if __name__ == "__main__":
    from multiprocessing import Process, freeze_support
    freeze_support()
    start(reload=False)

# Terminated -> Termination condition depends only on current action, not history of steps