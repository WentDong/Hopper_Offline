import gym
import numpy as np
import torch
import time

import json
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

from evaluate import *
from dataloader import SamaplesDataset
from args import get_args
from agents.BAIL import utils, bail_training
from agents.BAIL.mcret import *
from agents.BAIL.bail_training import Value, train_upper_envelope, plot_envelope, plot_envelope_with_clipping
from agents.BC.bc_agent import BC
from scripts.utils import plot_eval
from torch.utils.data import DataLoader
from tqdm import *
from torch.utils.tensorboard import SummaryWriter

# check directory
# print('data directory', os.getcwd())
# check pytorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("running on device:", device)
from utils import plot_eval

def get_mcret(replay_buffer, args):
    states, returns = get_mc(replay_buffer, args.data_name, args.gamma, args.rollout, args.augment_mc, args.device)

    return states, returns


def train_ue(states, returns, args):
    if not os.path.exists('%s/Stat_UE_%s.pth' % ("./agents/bail/checkpoints", args.setting_name + '_lok%s' % args.ue_loss_k + "_trunc%s" % args.trajectory_truncation if args.trajectory_truncation > 0 else "")):
        # train ue
        print('ue train starts --')
        print('with testing MClength:', args.rollout, 'training loss ratio k:', args.ue_loss_k)
        upper_envelope, _ = train_upper_envelope(states, returns, state_dim=11, upper_learning_rate=args.ue_lr,
                                                 weight_decay=args.ue_wd, num_epoches=args.ue_n_epochs, k=args.ue_loss_k)
        torch.save(upper_envelope.state_dict(), '%s/Stat_UE_%s.pth' % ("./agents/bail/checkpoints", args.setting_name + '_lok%s' % args.ue_loss_k + "_trunc%s" % args.trajectory_truncation if args.trajectory_truncation > 0 else ""))

        print('plotting ue --')
        plot_envelope(upper_envelope, states, returns, args.ue_setting, [args.ue_lr, args.ue_wd, args.ue_loss_k, args.ue_n_epochs, 4])

    else:
        upper_envelope = Value(state_dim=11, activation='relu')
        upper_envelope.load_state_dict(
            torch.load('%s/Stat_UE_%s.pth' % ("./agents/bail/checkpoints", args.setting_name + '_lok%s' % args.ue_loss_k + "_trunc%s" % args.trajectory_truncation if args.trajectory_truncation > 0 else "")))
        print('Load envelope with training loss ratio k:', args.ue_loss_k)

    # do clipping if needed
    C = plot_envelope_with_clipping(upper_envelope, states, returns, args.ue_setting,
                                    [args.ue_lr, args.ue_wd, args.ue_loss_k, args.max_timesteps, 4],
                                    S=args.detect_interval) if args.clip_ue else None
    print('clipping at:', C)

    return upper_envelope, C


def select_batch_ue(replay_buffer, states, returns, upper_envelope, C, args):
    states = torch.from_numpy(states).to(device)
    returns = torch.from_numpy(returns).to(device)
    
    upper_envelope = upper_envelope.to(device)

    ratios = []
    for i in range(states.shape[0]):
        s, ret = states[i], returns[i]
        s_val = upper_envelope(s.unsqueeze(dim=0).float()).detach().squeeze()
        ratios.append(ret / torch.min(s_val, C) if C is not None else ret / s_val)

    ratios = torch.stack(ratios).view(-1)
    increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
    bor_ind = increasing_ratio_indices[-int(args.select_percentage * states.shape[0])]
    border = ratios[bor_ind]

    '''begin selection'''
    selected_buffer = utils.ReplayBuffer()
    print('Selecting with ue border', border.item())
    for i in range(states.shape[0]):
        rat = ratios[i]
        if rat >= border:
            obs, next_obs, act, reward, done = replay_buffer.index(i)
            selected_buffer.add((obs, next_obs, act, reward, done))

    initial_len, selected_len = replay_buffer.get_length(), selected_buffer.get_length()
    print('border:', border, 'selecting ratio:', selected_len, '/', initial_len)

    return (selected_buffer, selected_len, border)


def train(model, dataLoader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    Mx_Reward = 0
    timesteps = 0
    writer = SummaryWriter()
    Eval = Evaluator(device=args.device)
    idx = 0
    dir = os.path.join(args.save_dir, "BAIL", str(idx))
    while os.path.exists(dir):
        idx += 1
        dir = os.path.join(args.save_dir, "BAIL", str(idx))
    os.makedirs(dir)
    with open(os.path.join(dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    Reward_log = []
    step = 0
    if args.plot:
        Reward_log = []
    
    for epoch in trange(args.n_epochs):
        with tqdm(total=len(dataLoader)) as pbar:
            for batch in dataLoader:
                # Get data
                state = batch["state"].float().to(args.device)
                # print(state.shape, type(state))
                action = batch["action"].float().to(args.device)
                loss = model.train(state, action)

                optimizer.zero_grad()
                writer.add_scalar("actor_loss", loss.item(), step)

                loss.backward()
                # Update parameters
                optimizer.step()
                if args.plot and ((timesteps + len(state)) // args.plot_interval) > timesteps//args.plot_interval:
                    Reward, _ = Eval.evaluate(model)
                    Reward_log.append(Reward)
                timesteps += len(state)
                pbar.set_description(f"Epoch: {epoch}, Timesteps: {timesteps}")
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                if timesteps > args.max_timesteps:
                    break
            # Print loss
        Reward, episodes_len = Eval.evaluate(model)
        if Reward > Mx_Reward:
            torch.save(model.state_dict(), os.path.join(dir, "BAIL_best.pth"))
            Mx_Reward = Reward
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(dir, "BAIL_{}.pth".format(epoch % 10)))
        # tqdm.set_description("Epoch: {}, Reward: {}".format(epoch, Reward))
        print("Epoch: {}, Timesteps: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, timesteps, Reward, episodes_len))
        with open(os.path.join(dir, "log.txt"), "a") as f:
            f.write("Epoch: {}, Timesteps: {}, Reward: {}, Mean Episodes Length: {}\n".format(epoch, timesteps, Reward, episodes_len))
        if timesteps > args.max_timesteps:
            break
    if args.plot:
        Reward_log = np.array(Reward_log)
        np.save(os.path.join(dir, "BAIL_reward.npy"), Reward_log)

        return Reward_log.tolist()


if __name__ == "__main__":
    args = get_args("bail")

    # prepare replay buffer
    replay_buffer = utils.ReplayBuffer()
    replay_buffer.load(args.dataset_path, args.file_name, args.trajectory_truncation)

    # get mc returns
    states, returns = get_mcret(replay_buffer, args)

    # train ue
    ue_model, C = train_ue(states, returns, args)

    # select batch
    selected_buffer, selected_len, border = select_batch_ue(replay_buffer, states, returns, ue_model, C, args)

    # prepare dataloader
    dataset = SamaplesDataset.from_buffer(selected_buffer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # train bc
    if not args.plot:
        model = BC(state_dim=11, action_dim=3, hidden_dim=args.hidden_dim).to(args.device)
        train(model, dataloader, args)
    else:
        Reward_logs = []

        for _ in range(args.training_iteration):
            model = BC(state_dim=11, action_dim=3, hidden_dim=args.hidden_dim).to(args.device)
            Reward_log = train(model, dataloader, args)
            Reward_logs.append(Reward_log)
        Reward_logs = np.array(Reward_logs)
        np.save(os.path.join(args.save_dir, "BAIL_Rewards.npy"), Reward_logs)
        plot_eval(args.plot_interval, Reward_logs, "BAIL")




