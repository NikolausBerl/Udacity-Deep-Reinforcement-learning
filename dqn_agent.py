import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size



BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, is_double_Q_network, is_experience_replay):
                 
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.is_double_Q_network = is_double_Q_network
        self.is_experience_replay = is_experience_replay
        self.test_on = False
        

        
            
        
        # if (self.is_double_Q_network == True):
        #   print("Used Algorithm: Double-Deep Q-Network")
        # else:
        #    print("Used Algorithm: Deep Q Network")
        

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

              

        if is_experience_replay == False:
            # Batch_SIZE = 1
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, 1, seed)
        else:   
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        
        self.memory.add(state, action, reward, next_state, done)
        
        
        if self.is_experience_replay == False:
            #  learn each Time_Step --> No experience replay, so sample has only one experience to choose.
            #  So, no sampling is possible
            
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)   
        else:
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        #print("Epsilon", eps)
        #print("vor Action-Rückgabe")
        # Epsilon-greedy action selection
        if random.random() > eps:
            #print("argmax-action")
            return np.argmax(action_values.cpu().data.numpy())  
        else:
            #print("Zufallswert")
            return random.choice(np.arange(self.action_size))
            

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        if self.is_double_Q_network:
            
            # Test-Parameter: With "test_on = True", content and Shape-Information are printed out.
            # "test_on" is a Agent-Class-Variable
            # Calculate the  Q-Values of the local Network for the next state.
                    
            Q_local_next_values = self.qnetwork_local(next_states).detach()
            if (self.test_on == True):
                #  Q_local_next_values.shape
                print("Q_local_next_values.shape", Q_local_next_values.shape)
                print("Q_local_next_values", Q_local_next_values)
            
            # Calculate the Index of the best Action of the above calculated Values of the local Network
            max_Q_local_next_index = torch.max(Q_local_next_values, 1)[1] 
            
            if (self.test_on == True):
                # max_Q_local_next_index.shape:
                print("max_Q_local_next_index.shape", max_Q_local_next_index.shape)
                print("max_Q_local_next_index", max_Q_local_next_index)
            
            # Calculate the  Q-Values of the target Network for the next state.
            Q_target_next_values = self.qnetwork_target(next_states).detach()
            if (self.test_on == True):
                # Q_target_next_values.shape: [64,4]
                print("Q_target_next_values.shape", Q_target_next_values.shape)
                print("Q_target_next_values", Q_target_next_values)
        
            # Take Index of local Network to choos action value of target network
            max_new_Q_values = Q_target_next_values.gather(1, max_Q_local_next_index.unsqueeze(1))
            if (self.test_on == True):
                # max_new_Q_values.shape:[64,1]
                print("max_new_Q_values.shape", max_new_Q_values.shape)
                print("max_new_Q_values", max_new_Q_values)
            '''
            # Here a second programming solution, but less understandable
            Q_targets_next_index = torch.argmax(self.qnetwork_local(next_states).detach(),1)
            max_new_Q_values = self.qnetwork_target(next_states).detach().gather(1,Q_targets_next_index.unsqueeze(1))
            '''
        else:
        # Get max predicted Q values (for next states) from target model
            max_new_Q_values = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * max_new_Q_values * (1 - dones))
        if (self.test_on == True):
            print("Q_targets.shape", Q_targets.shape)
            print("Q_targets", Q_targets)
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if (self.test_on == True):
            print("Q_expected.shape", Q_expected.shape)
            print("Q_expected", Q_expected)
            self.test_on = False
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        
            
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)