import os
import argparse
import numpy as np
import csv
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from env.carrot_env import CarrotEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import obs_as_tensor
import cv2

def visualize(env, model, seed, beta=0.95):
    N = 30 

    model_folder = 'results/sb3/seed_{}'.format(seed)

    # init obs
    obs = env.reset()

    imgs_mpc_record = np.zeros((N, 256, 256))

    reward_episode = []
    for j in range(N):
        imgs_mpc_record[j] = env.render()
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_episode.append(reward)
    
    R = 0
    for i, r in enumerate(reward_episode[::-1]):
        R = r + beta * R

    print("Return: ", R)

    path = os.path.join(model_folder, 'video')
    os.system('mkdir -p ' + path)

    video_path = path + '.avi'
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    print('Save video as %s' % video_path)
    out = cv2.VideoWriter(video_path, fourcc, 3, (256, 256))


    for i in range(imgs_mpc_record.shape[0]):
        img = np.zeros((256, 256, 3)).astype(np.uint8)
        img[:, :, :] = imgs_mpc_record[i, :, :, None]
        out.write(img)

        cv2.imwrite(os.path.join(path, '%d.png' % i), img)

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
    parser.add_argument("--A2C", action="store_true")
    args = parser.parse_args()
    # set device to cpu or cuda
    device = torch.device('cpu')

    if(torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    env = CarrotEnv()

    model_dir = "model/H/"
    if args.A2C:
        model = A2C.load(model_dir + "A2C_{}".format(args.seed), env=env)
    else:
        model = PPO.load(model_dir + "PPO_{}".format(args.seed), env=env)

    visualize(env, model, args.seed)