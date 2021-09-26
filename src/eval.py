import os
import argparse
import numpy as np
import csv
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from env.carrot_env import CarrotEnv

from PPO import ActorCritic, CompressionCNN
import cv2


def visualize(env, nz, seed, beta=0.95):
    N = 30

    env_name = "Carrot"   #"Carrot_circle"
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    action_dim = env.action_space.n
    random_seed = seed
    action_std = 0.6
    nf = 64

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, nz)
    compression_path = directory + "compression_{}_{}_{}.pth".format(env_name, random_seed, nz)

    policy = ActorCritic(nz, action_dim, False, action_std).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    compression = CompressionCNN(nf, nz).to(device)
    compression.load_state_dict(torch.load(compression_path, map_location=lambda storage, loc: storage))

    model_folder = 'results/nz_{}_seed_{}/'.format(nz, seed) + env_name +"/"

    # init obs
    obs = env.reset()

    imgs_mpc_record = np.zeros((N, 256, 256))

    reward_episode = []
    for j in range(N):
        imgs_mpc_record[j] = env.render()

        state = torch.FloatTensor(obs).to(device).reshape(1, 3, 32, 32)
        ais = compression(state).detach()
        action, _ = policy.act(ais)
        action = action.item()
        obs, reward, _, _ = env.step(action)
        reward_episode.append(reward)

    rets = []
    R = 0
    for i, r in enumerate(reward_episode[::-1]):
        R = r + beta * R
        rets.insert(0, R)

    print("Return: ", rets[0])

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
    parser.add_argument("--nz", type=int, default=128)
    args = parser.parse_args()
    # set device to cpu or cuda
    device = torch.device('cpu')

    if (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    env = CarrotEnv()

    visualize(env, args.nz, args.seed)
