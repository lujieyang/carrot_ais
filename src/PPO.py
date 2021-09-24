import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical



################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.obs = []
        self.u_onehot = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.obs[:]
        del self.u_onehot[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class CompressionCNN(nn.Module):
    """
    3.3 Implements a compression function y -> z. Is a more efficient (translation-invariant)
    version of the CompressionMLP.
    """
    def __init__(self, nf, nz):
        super(CompressionCNN, self).__init__()
        input_dim = 3
        # Input image: 3 x 32 x 32
        self.compression_conv = nn.Sequential(
            nn.Conv2d(input_dim, nf, 3, 1, 1),  # 32 x 32
            nn.LeakyReLU(),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),     # 16 x 16
            nn.LeakyReLU(),
            nn.Conv2d(nf * 2, nf * 2, 3, 2, 1), # 8 x 8
            nn.LeakyReLU(),
            nn.Conv2d(nf * 2, nf * 2, 3, 2, 1), # 4 x 4
            nn.LeakyReLU(),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), # 4 x 4
            nn.LeakyReLU()
        )

        # after compressing & flattening, pass it through another layer of mlps
        # to get the desired dimension of z.
        self.compression_conv_mlp = nn.Sequential(
            nn.Linear(4 * 4 * nf * 2, nf * 2),
            nn.ReLU(),
            nn.Linear(nf * 2, nf),
            nn.ReLU(),
            nn.Linear(nf, nz)
        )

    def forward(self, x):
        # input is B x 3 x 32 x 32 image.
        b = x.shape[0] # this is the batch size.
        x = self.compression_conv(x)
        x = x.view(b, -1)
        x = self.compression_conv_mlp(x)
        return x


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, lr_ais, gamma, lmbda, nf, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.lmbda = lmbda
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_dim = action_dim
        
        self.buffer = RolloutBuffer()

        self.compression = CompressionCNN(nf, state_dim).to(device)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, nf), nn.ReLU(),
            nn.Linear(nf, 2 * nf), nn.ReLU(),
            nn.Linear(2 * nf, 2 * nf), nn.ReLU(),
            nn.Linear(2 * nf, nf), nn.ReLU(),
            nn.Linear(nf, state_dim)).to(device)
        self.reward = nn.Sequential(
            nn.Linear(state_dim + action_dim, nf), nn.ReLU(),
            nn.Linear(nf, 2 * nf), nn.ReLU(),
            nn.Linear(2 * nf, 2 * nf), nn.ReLU(),
            nn.Linear(2 * nf, nf), nn.ReLU(),
            nn.Linear(nf, 1)).to(device)
        ais_param = list(self.compression.parameters()) + list(self.dynamics.parameters()) + list(self.reward.parameters())
        self.ais_optimizer = torch.optim.Adam(ais_param, lr=lr_ais, betas=(0.9, 0.999))


        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).reshape(1, 3, 32, 32)
                # WARNING: detach the AIS
                ais = self.compression(state).detach()
                action, action_logprob = self.policy_old.act(ais)
            
            self.buffer.states.append(ais)
            self.buffer.obs.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            u_onehot = torch.FloatTensor(self.action_dim).to(device)
            u_onehot.zero_()
            u_onehot.scatter_(0, action, 1)
            self.buffer.u_onehot.append(u_onehot)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        obs = torch.squeeze(torch.stack(self.buffer.obs, dim=0)).detach().to(device)
        u_onehot = torch.squeeze(torch.stack(self.buffer.u_onehot[:-1], dim=0)).detach().to(device)
        B = rewards.shape[0]-1
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # AIS loss
            z_i = self.compression(obs[:-1])
            z_f = self.compression(obs[1:])

            rhat = self.reward(torch.cat((z_i, u_onehot), 1))
            zhat_f = self.dynamics(torch.cat((z_i, u_onehot), 1))

            r_loss = self.MseLoss(rhat, rewards[:-1].view(B, 1))
            z_loss = self.MseLoss(zhat_f, z_f)

            ais_loss = self.lmbda * r_loss + (1 - self.lmbda) * z_loss

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            self.ais_optimizer.zero_grad()
            ais_loss.backward()
            self.ais_optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
        print('r loss: {}, z loss: {}, policy loss: {}'.format(r_loss, z_loss, loss.mean()))
    
    def save(self, checkpoint_path, compression_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(self.compression.state_dict(), compression_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


