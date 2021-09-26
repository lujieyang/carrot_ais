import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from env.carrot_env import CarrotEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
    parser.add_argument("--A2C", action="store_true")
    args = parser.parse_args()

    def env_maker(rank):
        def hof():
            env = CarrotEnv()
            env = Monitor(env)
            env.seed(args.seed + rank)
            return env

        return hof

    num_envs = 16
    env = SubprocVecEnv([env_maker(rank) for rank in range(num_envs)])
    env = VecNormalize(env, norm_obs=False)

    model_dir = "model/H/"
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    if args.A2C:
        model = A2C("MlpPolicy", env, verbose=1, seed=args.seed)
        model.learn(total_timesteps=int(5e4))
        model.save(model_dir + "A2C_{}".format(args.seed))
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=512)
        model.learn(total_timesteps=int(4e5))
        model.save(model_dir + "PPO_{}".format(args.seed))

