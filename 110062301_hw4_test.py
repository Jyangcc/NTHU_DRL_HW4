import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Layer_norm(nn.Module):
    def __init__(self, in_features, out_features):
        # Layer, Norm, Afn, Drop, Residual
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
        self.norm = nn.LayerNorm(out_features)
       
    def forward(self, layer_in):
        out = self.fc(layer_in)
        out = self.norm(out)
        
        return out

scale = torch.zeros(22) + 0.5
bias  = torch.zeros(22) + 0.5



class ActorNetwork(nn.Module):
    def __init__(self, action_dim = 22, min_log_sigma = -20, max_log_sigma = 2):
        super(ActorNetwork, self).__init__()
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        
        self.v_tgt1 = Layer_norm(242, 512)
        self.v_tgt2 = Layer_norm(512, 256)
        
        
        self.body1 = Layer_norm(97, 512)
        self.body2 = Layer_norm(512, 256)
        
        
        self.mean1 = nn.Linear(512, 64)
        self.mean2 = nn.Linear(64, action_dim)
        
        
        
        
    def forward(self, x):
        if(x.dim() == 1):
            x = x.reshape(1,-1)
        
        
        v_tgt = x[:, :242]
        body = x[:, 242:]
        
        
        v_tgt_out = F.elu(self.v_tgt1(v_tgt))
        v_tgt_out = F.elu(self.v_tgt2(v_tgt_out))
        
        
        body_out = F.elu(self.body1(body))
        body_out = F.elu(self.body2(body_out))
        
        concat = torch.cat([v_tgt_out, body_out], dim=1)
        
        
        mean = F.elu(self.mean1(concat))
        mean = torch.tanh(self.mean2(mean)) * scale + bias
        
        return mean 
        



################################
##     state Preprocessing    ##
################################


def Skip_frame_Step(env, action, skip_frame = 4):
    total_reward = 0.0
    done = False
    for i in range(skip_frame):
        state, reward, done, info  = env.step(action)
        
        total_reward += reward
        if done:
            break
    
    return state, total_reward, done, info





class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)
        
        self.actor = ActorNetwork()
        
        checkpoint = torch.load('110062301_hw4_data')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
        self.step = 0
        self.last_action = None

    def act(self, observation):
        with torch.no_grad():
            if self.step % 4 == 0:
                self.actor.eval()
                action = self.actor(observation)
                
            
                action = action.detach().cpu()
                
                action = torch.clamp(action, 0.0, 1.0)
                self.last_action = action.numpy()[0]
            
            self.step += 1
            
            return self.last_action
        
