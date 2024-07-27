import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

LOG_STD_MAX = 3
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, args,use_xavier = True):
        super().__init__()
        self.args = args
        input_shape = args.action_shape
        output_shape = args.obs_shape
        self.high_action = args.high_action
        self.low_action = args.low_action
            
        self.fc1 = nn.Linear(input_shape , 16*input_shape)
        self.fc2 = nn.Linear(16*input_shape, 16*input_shape)
        self.fc_mean = nn.Linear(input_shape*16, output_shape)
        self.fc_logstd = nn.Linear(16*input_shape, output_shape)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((self.high_action - self.low_action ) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((self.high_action + self.low_action ) / 2.0, dtype=torch.float32)
        )
        if use_xavier:
            self._initialize_weights()
            
    def forward(self,x):     
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

class SoftQNetwork(nn.Module):
    def __init__(self, args,use_xavier = True):
        super().__init__()
        self.args = args
        input_shape = args.action_shape
        output_shape = args.obs_shape

        self.fc1 = nn.Linear(input_shape+output_shape , 16*input_shape)
        self.fc2 = nn.Linear(16*input_shape, 16*input_shape)
        self.fc3 = nn.Linear(16*input_shape, 1)

        if use_xavier:
            self._initialize_weights()

    def forward(self, x, actions):
        x = torch.cat([x, actions], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
