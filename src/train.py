
import os
import torch
import numpy as np
from tensordict import TensorDict
import pickle
from torch import distributions
from scipy.optimize import linear_sum_assignment
from Codebook_Learning_RL.DataPrep import dataPrep
from Codebook_Learning_RL.env_ddpg import envCB
from Codebook_Learning_RL.clustering import KMeans_only
from Codebook_Learning_RL.function_lib import bf_gain_cal, corr_mining
import time
import re
import copy
import random
import yaml
from tqdm.auto import tqdm
from datetime import datetime
from sac.sac_algorithm import SAC
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def make_dirs_and_open(file_path, mode):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, mode)


def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)


def getdatetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



def train(env,options,train_options,agent,beam_id,writer):

    device = train_options['device']
    CB_Env = env   
    if train_options['overall_iter'] == 1:
        state = torch.zeros((1, options['num_ant'])).float().to(device)
        print('Initial State Activated.')
    else:
        state = train_options['state']
    
    
    #training_loop
    iteration = 0
    num_of_iter = train_options['num_iter']  
    while iteration < num_of_iter:
          
        action = agent.act_train(state)
        reward_pred, bf_gain_pred, action_quant_pred, state_1_pred = CB_Env.get_reward(action.to(device))
        reward_pred = torch.from_numpy(reward_pred).float().to(device)
        
        action_pred_noisy = action
        mat_dist = torch.abs(action_pred_noisy.reshape(options['num_ant'], 1) - options['ph_table_rep'])
        action_quant = options['ph_table_rep'][range(options['num_ant']), torch.argmin(mat_dist, dim=1)].reshape(1, -1)
   
        state_1, reward, bf_gain, terminal = CB_Env.step(action_quant)
        reward = torch.from_numpy(reward).float().to(device)
        action = action_quant.reshape((1, -1)).float().to(device)

        batch_transition = TensorDict(
            {
                "observations":state.detach().clone(),
                "next_observations":state_1.detach().clone(),
                "actions":action.detach().clone(),
                "rewards":reward.detach().clone(),
                "dones":torch.tensor([terminal]*state.shape[0]),
            },
            batch_size = [state.shape[0]],
        )
        agent.rb.extend(batch_transition)

        agent.observe(
            state.detach().clone(), 
            action_quant_pred.detach().clone(), 
            reward_pred.detach().clone(),
            state_1_pred.detach().clone(),
            torch.tensor([terminal]*state.shape[0]), 
           )

        
        iteration += 1
        train_options['overall_iter'] += 1
        state = state_1
        
        new_gain = torch.Tensor.cpu(CB_Env.achievement).detach().numpy().reshape((1, 1))
        max_previous_gain = max(CB_Env.gain_history)
        if new_gain > max_previous_gain:
            CB_Env.gain_history.append(float(new_gain[0][0]))                   
        else:
            CB_Env.gain_history.append(float(max_previous_gain[0][0]))
            
    train_options['state'] = state  # used for the next loop
    train_options['best_state'] = CB_Env.best_bf_vec  # used for clustering and assignment
    if (train_options['overall_iter']-1)%500==0:
        writer.add_scalar(f'Beamforming_gain_beam_{beam_id}', 
                          {'gain':max_previous_gain,'EGC':CB_Env.compute_EGC()}, 
                                                               train_options['overall_iter'])
        # print(
        #     "Beam: %d, Iter: %d, Reward pred: %d, Reward: %d, BF Gain pred: %.2f, BF Gain: %.2f" % \
        #     (beam_id, train_options['overall_iter'],
        #      int(torch.Tensor.cpu(reward_pred).numpy().squeeze()),
        #      int(torch.Tensor.cpu(reward).numpy().squeeze()),
        #      torch.Tensor.cpu(bf_gain_pred.detach()).numpy().squeeze(),
        #      torch.Tensor.cpu(bf_gain.detach()).numpy().squeeze(),))      
    
    return train_options


def main():
    experiments_dir = 'experiments/runs/'
    env_config_path = 'experiments/configs/env_config.yaml'
    sac_config_path = 'experiments/configs/sac_config.yaml'
    with open(sac_config_path) as f:
        train_opt=yaml.load(f, Loader=yaml.FullLoader)

    with open(env_config_path) as f:
        options =yaml.load(f, Loader=yaml.FullLoader)

    train_opt['high_action'] = torch.pi
    train_opt['low_action'] = -torch.pi
    train_opt['action_shape'] = options['num_ant']
    train_opt['obs_shape'] = options['num_ant']


    sac_config = Config(train_opt)
    env_config = Config(options)

    exp_name = sanitize_filename(sac_config.exp_name)
    run_name = f"{exp_name}__{getdatetime()}"
    run_name = run_name[:255]
    run_dir = os.path.join(experiments_dir, run_name) 
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "SAC_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(sac_config).items()])),
    )
    writer.add_text(
        "env_parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(env_config).items()])),
    )

    random.seed(sac_config.seed)
    np.random.seed(sac_config.seed)
    torch.manual_seed(sac_config.seed)
    torch.backends.cudnn.deterministic = sac_config.torch_deterministic 
    torch.autograd.set_detect_anomaly(True)

    with make_dirs_and_open(os.path.join(run_dir, 'configs/sac_config.yaml'), 'w') as f:
        yaml.dump(train_opt, f, indent=4, default_flow_style=False)
   
    with make_dirs_and_open(os.path.join(run_dir, 'configs/env_config.yaml'), 'w') as f:
        yaml.dump(options, f, indent=4, default_flow_style=False)

    if not os.path.exists(os.path.join(run_dir, 'beams/')):
        os.mkdir(os.path.join(run_dir, 'beams/')) # to store the beamforming codebooks
    if not os.path.exists(os.path.join(run_dir, 'beamforming_gain_records/')):
        os.mkdir(os.path.join(run_dir, 'beamforming_gain_records/')) # to store the beamforming gain records

    ch = dataPrep(options['path'])
    ch = np.concatenate((ch[:, :options['num_ant']],
                        ch[:, int(ch.shape[1] / 2):int(ch.shape[1] / 2) + options['num_ant']]), axis=1)
    with torch.cuda.device(options['gpu_idx']):
        u_classifier, sensing_beam = KMeans_only(ch, options['num_NNs'], n_bit=options['num_bits'], n_rand_beam=30)
        np.save(os.path.join(run_dir,'sensing_beam.npy'), sensing_beam)
        sensing_beam = torch.from_numpy(sensing_beam).float().cuda()

        filename =  os.path.join(run_dir, 'kmeans_model.sav')
        pickle.dump(u_classifier, open(filename, 'wb'))

        # Quantization settings
        options['num_ph'] = 2 ** options['num_bits']
        options['multi_step'] = torch.from_numpy(
            np.linspace(int(-(options['num_ph'] - 2) / 2),
                        int(options['num_ph'] / 2),
                        num=options['num_ph'],
                        endpoint=True)).type(dtype=torch.float32).reshape(1, -1).cuda()
        options['pi'] = torch.tensor(np.pi).cuda()
        options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
        options['ph_table'].cuda()
        options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)
        env_list = []
        train_opt_list = []
        agent_list=[]
    
    for beam_id in range(options['num_NNs']):
        env_list.append(envCB(ch, options['num_ant'], options['num_bits'], beam_id, options, run_dir))
        train_opt_list.append(copy.deepcopy(train_opt))
        agent = SAC(sac_config,writer)
        agent.start_time = time.time()
        agent_list.append(agent)

    with torch.cuda.device(options['gpu_idx']):
        for sample_id in range(options['num_loop']):

            # ---------- Sampling ---------- #
            n_sample = int(ch.shape[0] * options['ch_sample_ratio'])
            ch_sample_id = np.random.permutation(ch.shape[0])[0:n_sample]
            ch_sample = torch.from_numpy(ch[ch_sample_id, :]).float().cuda()

            # ---------- Clustering ---------- #
        #     start_time = time.time()

            bf_mat_sample = bf_gain_cal(sensing_beam, ch_sample)
            # print("Clustering -1 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()
            f_matrix = corr_mining(bf_mat_sample)
            f_matrix_np = torch.Tensor.cpu(f_matrix).numpy()
            # print("Clustering 0 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()
            labels = u_classifier.predict(np.transpose(f_matrix_np).astype(float))

            # print("Clustering 1 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()

            user_group = []  # order: clusters
            ch_group = []  # order: clusters
            for ii in range(options['num_NNs']):
                user_group.append(np.where(labels == ii)[0].tolist())
                ch_group.append(ch_sample[user_group[ii], :])

        #     print("Clustering 2 uses %s seconds." % (time.time() - start_time))

            # ---------- Assignment ---------- #
        #     start_time = time.time()

            # best_state matrix
            best_beam_mtx = torch.zeros((options['num_NNs'], 2 * options['num_ant'])).float().cuda()
            for pp in range(options['num_NNs']):
                best_beam_mtx[pp, :] = env_list[pp].best_bf_vec
            gain_mtx = bf_gain_cal(best_beam_mtx, ch_sample)  # (n_beam, n_user)
            for ii in range(options['num_NNs']):
                if ii == 0:
                    cost_mtx = torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options['num_NNs'], -1)
                else:
                    sub = torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options['num_NNs'], -1)
                    cost_mtx = torch.cat((cost_mtx, sub), dim=1)
            cost_mtx = -torch.Tensor.cpu(cost_mtx).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_mtx)
            assignment_record = dict(zip(row_ind.tolist(), col_ind.tolist()))  # key: network, value: cluster
            for ii in range(options['num_NNs']):
                env_list[ii].ch = ch_group[assignment_record[ii]]
                env_list[ii].EGC_history.append(env_list[ii].compute_EGC())
        #     print("Assignment uses %s seconds." % (time.time() - start_time))
            for beam_id in range(options['num_NNs']):
                train_opt_list[beam_id] = train(env_list[beam_id],options, train_opt_list[beam_id],agent_list[beam_id], beam_id,writer)

    writer.close()
    num_beam = options['num_NNs']
    num_ant = options['num_ant']
    for beam_id in range(num_beam):
        fname = 'beams_' + str(beam_id) + '_max.txt'
        with open(os.path.join(run_dir,'beams',fname), 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            results[beam_id, :] = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)

    results = (1 / np.sqrt(num_ant)) * (results[:, ::2] + 1j * results[:, 1::2])

    scio.savemat(os.path.join(run_dir,'beams','beam_codebook.mat'), {'beams': results})
    for beam_id  in range(num_beam):
        EGC = env_list[beam_id].compute_EGC()
        gain_record = env_list[beam_id].gain_history[1:].append(EGC)
        gain_record = np.array(gain_record)/options['num_ant']
        np.save(os.path.join(run_dir,'beamforming_gain_records',f'beam_{beam_id}_gain_records'),gain_record)

if __name__ == '__main__':
    main()