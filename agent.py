from net import ActorNet, CriticNet
from collections import deque
import random
import torch
import copy
import pickle
import lz4.frame
import psutil

class Agent:
    def __init__(self, ntime=4):
        self.actor = ActorNet(ntime).cuda()
        self.critic = CriticNet(ntime).cuda()
        self.state_buffer = deque(maxlen=ntime)
        self.replay_buffer = deque(maxlen=1000)
        self.init_buffer = True

        self.gamma = 0.97
        self.ntime = ntime

    def get_action(self, state):
        state_avg = (torch.sum(state, dim=1, keepdim=True)) / self.ntime

        if self.init_buffer:
            #State buffer initialization, fill up state buffer.
            for i in range(self.state_buffer.maxlen):
                self.state_buffer.append(state_avg)
            self.init_buffer = False

        self.state_buffer.append(state_avg)
        
        states = torch.cat(list(self.state_buffer), dim=1)
        states = torch.cat([state, states], dim=1).cuda()

        action = torch.zeros_like(state).cuda()
        acc_stop = random.randint(2,4)
        for i in range(acc_stop):
            action = self.actor(states, action)
        return action

    def remember(self, state, action, reward, next_state):
        compressed = lz4.frame.compress(pickle.dumps([state, action, reward, next_state, self.state_buffer]))
        # print(psutil.virtual_memory().used / 1e9)
        self.replay_buffer.append(compressed)

    def replay(self):
        memory = random.sample(self.replay_buffer, k=1)
        memory = [pickle.loads(lz4.frame.decompress(memory[0]))]

        for (state, action, reward, next_state, state_buffer) in memory:
            next_state_avg = 1 - (3 - torch.sum(next_state, dim=1, keepdim=True))
            next_state_buffer = copy.deepcopy(state_buffer)
            next_state_buffer.append(next_state_avg)


            reward = torch.tensor([[reward]])

            states = torch.cat(list(state_buffer), dim=1)
            states = torch.cat([state, states], dim=1)

            next_states = torch.cat(list(next_state_buffer), dim=1)
            next_states = torch.cat([next_state, next_states], dim=1)

            

            action_hat = self.actor(states.cuda(), torch.zeros_like(action).cuda())
            actor_loss = self.critic(states.cuda(), action_hat.cuda()) + torch.nn.functional.mse_loss(action_hat, action.cuda())
            self.actor.backward(actor_loss)

            critique_hat = self.critic(states.detach().cuda(), action.detach().cuda())
            next_critique_hat = self.gamma * self.critic(next_states.detach().cuda(), self.actor(states.detach().cuda(), torch.zeros_like(action).cuda()))
            critic_loss = torch.nn.functional.mse_loss(critique_hat, reward.cuda()+next_critique_hat)
            self.critic.backward(critic_loss)