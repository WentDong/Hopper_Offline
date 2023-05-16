import gym
import numpy as np
import torch
import argparse
import os
from agents.bail import utils


def get_mc(replay_buffer, data_name, gamma=0.99, rollout=1000, augment_mc='gain', device=None):
    print('MClength:', rollout)
    print('Discount value', gamma)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)

    result_dir = "agents/bail/results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    buffer_name = replay_buffer.name
    print("---------------------------------------")
    print("Settings: " + data_name)
    print("---------------------------------------")

    print('Starting MC calculation, type:', augment_mc)

    if augment_mc == 'gain':
        states, gains = calculate_mc_gain(replay_buffer, rollout=rollout, gamma=gamma)

        if not os.path.exists(os.path.join(result_dir, 'ueMC_%s_S.npy' % buffer_name)):
            np.save(os.path.join(result_dir, 'ueMC_%s_S' % buffer_name), states)
        print(len(gains))
        np.save(os.path.join(result_dir, 'ueMC_%s' % data_name, gains))
    else:
        raise Exception('! undefined mc calculation type')

    print('Calculation finished ==')
    if isinstance(states, list):
        states = np.stack(states)
    if isinstance(gains, list):
        gains = np.stack(gains)
    return states, gains


def calculate_mc_gain(replay_buffer, rollout=1000, gamma=0.99):
    states, actions, gts, endpoint, dist = calculate_mc_return_no_aug(replay_buffer, gamma)

    aug_gts = gts[:]

    # Add augmentation terms
    start = 0
    for i in range(len(endpoint)):
        end = endpoint[i]
        if end - start < rollout - 1:
            # Early terminated episodes
            start = end + 1
            continue

        # episodes not early terminated
        for j in range(end, start, -1):
            interval = dist[start: start + end - j + 2]
            index = interval.index(min(interval))
            # term = end - j + 1
            # term += rollout - index
            aug_gts[j] += gamma ** (end - j + 1) * gts[start + index]
            if index != end - j + 1:
                aug_gts[j] -= gamma ** (rollout) * gts[index + j]
                # term -= end - index - j + 1

            # print("number of terms used to calculate mc ret: ", term)
        start = end + 1

    return states, aug_gts


def calculate_mc_return_no_aug(replaybuffer, gamma=0.99):
    """
    Calculate the MC return without augmentation
    Input: replaybuffer: BCQ replay buffer
    Output: states, actions, returns (no aug)
    """

    gts = []
    states = []
    actions = []

    g = 0

    g = 0
    prev_s = 0
    termination_point = 0

    endpoint = []
    dist = []  # L2 distance between the current state and the termination point

    length = replaybuffer.get_length()

    for ind in range(length - 1, -1, -1):
        state, o2, action, r, done = replaybuffer.index(ind)

        states.append(state)
        actions.append(action)

        if done:
            g = r
            gts.append(g)
            endpoint.append(ind)
            termination_point = state
            prev_s = state
            dist.append(0)
            continue

        if np.array_equal(prev_s, o2):
            g = gamma * g + r
            prev_s = state
            dist.append(np.linalg.norm(state - termination_point))
        else:
            g = r
            endpoint.append(ind)
            termination_point = state
            prev_s = state
            dist.append(0)

        gts.append(g)

    return states[::-1], actions[::-1], gts[::-1], endpoint[::-1], dist[::-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_set", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--rollout", default=1000, type=int)
    args = parser.parse_args()

    exp_name = 'placeholder_mclen%s_gamma%s' % (args.rollout, args.gamma)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    get_mc(env_set=args.env_set, seed=args.seed,
           gamma=args.gamma, rollout=args.rollout)
