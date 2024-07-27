
import os
import torch
import numpy as np
from tensordict import TensorDict
def train(env,options,train_options,agent,beam_id):
#     action_pred_noisy = ounoise.get_action(action_pred,
#                                                t=train_options['overall_iter'])
    device_index = train_options['gpu']
    device = f'cuda:{device_index}'
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
        
        device_index = train_options['gpu']
        device = f'cuda:{device_index}'
        
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
                "observations":torch.from_numpy(state).clone(),
                "next_observations":torch.from_numpy(state_1).clone(),
                "actions":action.clone(),
                "rewards":torch.from_numpy(reward).clone(),
                "dones":torch.tensor([terminal]),
            },
            batch_size = [state.shape[0]],
        )
        agent.rb.extend(batch_transition)

        agent.observe(
            torch.from_numpy(state), 
            torch.from_numpy(action_quant_pred), 
            torch.from_numpy(reward_pred),
            torch.from_numpy(state_1_pred),
            torch.tensor([terminal]), 
           )

        
        iteration += 1
        train_options['overall_iter'] += 1
        state = state_1
        
        new_gain = torch.Tensor.cpu(CB_Env.achievement).detach().numpy().reshape((1, 1))
        max_previous_gain = max(CB_Env.gain_history)
        if new_gain > max_previous_gain:
            CB_Env.gain_history.append(float(new_gain))                   
        else:
            CB_Env.gain_history.append(float(max_previous_gain))
            
    train_options['state'] = state  # used for the next loop
    train_options['best_state'] = CB_Env.best_bf_vec  # used for clustering and assignment
    if (train_options['overall_iter']-1)%500==0:

        print(
            "Beam: %d, Iter: %d, Reward pred: %d, Reward: %d, BF Gain pred: %.2f, BF Gain: %.2f" % \
            (beam_id, train_options['overall_iter'],
             int(torch.Tensor.cpu(reward_pred).numpy().squeeze()),
             int(torch.Tensor.cpu(reward).numpy().squeeze()),
             torch.Tensor.cpu(bf_gain_pred.detach()).numpy().squeeze(),
             torch.Tensor.cpu(bf_gain.detach()).numpy().squeeze(),))      
    
    return train_options


def main():
    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    ch = dataPrep(options['path'])
    ch = np.concatenate((ch[:, :options['num_ant']],
                        ch[:, int(ch.shape[1] / 2):int(ch.shape[1] / 2) + options['num_ant']]), axis=1)
    with torch.cuda.device(options['gpu_idx']):
        u_classifier, sensing_beam = KMeans_only(ch, options['num_NNs'], n_bit=options['num_bits'], n_rand_beam=30)
        np.save('sensing_beam.npy', sensing_beam)
        sensing_beam = torch.from_numpy(sensing_beam).float().cuda()

        filename = 'kmeans_model.sav'
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
        actor_net_list = []
        critic_net_list = []
        actor_net_t_list = []
        critic_net_t_list = []
        ounoise_list = []
        env_list = []
        train_opt_list = []
        agent_list=[]

if __name__ == '__main__':
    main()