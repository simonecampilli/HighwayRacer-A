import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Registra manualmente l'ambiente
gym.register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

# Definisci la rete neurale
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Parametri
env = gym.make('highway-v0', render_mode='human')
input_dim = np.prod(env.observation_space.shape)  # Assicurati che questa dimensione sia corretta per l'ambiente
output_dim = env.action_space.n
lr = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
memory = deque(maxlen=2000)
batch_size = 64

# Inizializza il modello e l'ottimizzatore
model = DQN(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Funzione di preelaborazione per gestire le osservazioni
def preprocess_observation(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.array(obs).flatten()

# Funzione per selezionare l'azione
def select_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return model(state).argmax().item()

# Funzione per addestrare il modello
def train():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Converti le osservazioni utilizzando la funzione di preelaborazione
    states = torch.FloatTensor(np.array([preprocess_observation(s) for s in states]))
    next_states = torch.FloatTensor(np.array([preprocess_observation(ns) for ns in next_states]))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = model(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    loss = criterion(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Ciclo di addestramento
for episode in range(1000):
    state, _ = env.reset()  # Estrai l'osservazione corretta
    state = preprocess_observation(state)
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated  # Considera l'episodio terminato se `done` o `truncated` sono veri
        next_state = preprocess_observation(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Renderizza l'ambiente ogni 10 passi
        if step % 10 == 0:
            env.render()

        train()
        step += 1

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
