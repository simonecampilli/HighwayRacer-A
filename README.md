# Progetto DQN con HighwayEnv

Questo progetto implementa un agente di apprendimento per rinforzo utilizzando una rete neurale profonda (Deep Q-Network, DQN) per interagire con l'ambiente `highway-v0` fornito dalla libreria `highway-env`. L'agente viene addestrato per prendere decisioni ottimali in un ambiente di guida autostradale.

## Requisiti

Assicurati di avere i seguenti pacchetti installati nel tuo ambiente di sviluppo:

- `gymnasium`
- `highway-env`
- `numpy`
- `torch`

## Struttura del Progetto

- `main.py`: Contiene il codice principale per l'addestramento dell'agente DQN.
- `DQN`: Classe definita per la rete neurale profonda utilizzata dall'agente.
- `train`: Funzione per addestrare l'agente.
- `select_action`: Funzione per selezionare l'azione basata su una politica epsilon-greedy.
- `preprocess_observation`: Funzione per pre-elaborare le osservazioni dell'ambiente.

## Esecuzione del Progetto

Durante l'esecuzione, l'agente interagirà con l'ambiente `highway-v0` e verrà addestrato attraverso episodi. Ad ogni episodio, verrà stampata una riga che indica il totale del reward accumulato.

## Parametri

Di seguito sono elencati i principali parametri utilizzati nell'algoritmo DQN:

- `lr` (Learning Rate): Tasso di apprendimento per l'ottimizzatore.
- `gamma` (Discount Factor): Fattore di sconto per il calcolo del valore atteso.
- `epsilon` (Exploration Rate): Tasso di esplorazione iniziale.
- `epsilon_decay`: Fattore di decadimento del tasso di esplorazione.
- `epsilon_min`: Valore minimo per il tasso di esplorazione.
- `memory`: Dimensione della memoria di replay.
- `batch_size`: Dimensione del batch per l'addestramento.

## Funzioni Principali

### DQN

La classe `DQN` definisce la struttura della rete neurale profonda utilizzata per approssimare la funzione Q. È composta da tre livelli completamente connessi (fully connected layers).

### preprocess_observation

Questa funzione pre-elabora le osservazioni dell'ambiente per renderle adatte all'ingresso della rete neurale.

### select_action

Questa funzione seleziona un'azione basata su una politica epsilon-greedy, bilanciando l'esplorazione e lo sfruttamento.

### train

Questa funzione esegue il processo di addestramento della rete neurale utilizzando campioni casuali dalla memoria di replay.

---

# DQN Project with HighwayEnv

This project implements a reinforcement learning agent using a Deep Q-Network (DQN) to interact with the `highway-v0` environment provided by the `highway-env` library. The agent is trained to make optimal decisions in a highway driving environment.

## Requirements

Ensure you have the following packages installed in your development environment:

- `gymnasium`
- `highway-env`
- `numpy`
- `torch`

## Project Structure

- `main.py`: Contains the main code for training the DQN agent.
- `DQN`: Class defining the deep neural network used by the agent.
- `train`: Function to train the agent.
- `select_action`: Function to select actions based on an epsilon-greedy policy.
- `preprocess_observation`: Function to preprocess the environment observations.

## Running the Project

During execution, the agent will interact with the `highway-v0` environment and be trained through episodes. For each episode, a line indicating the total reward accumulated will be printed.

## Parameters

Below are the main parameters used in the DQN algorithm:

- `lr` (Learning Rate): Learning rate for the optimizer.
- `gamma` (Discount Factor): Discount factor for the expected value calculation.
- `epsilon` (Exploration Rate): Initial exploration rate.
- `epsilon_decay`: Decay factor for the exploration rate.
- `epsilon_min`: Minimum value for the exploration rate.
- `memory`: Size of the replay memory.
- `batch_size`: Batch size for training.

## Main Functions

### DQN

The `DQN` class defines the structure of the deep neural network used to approximate the Q-function. It consists of three fully connected layers.

### preprocess_observation

This function preprocesses the environment observations to make them suitable for input into the neural network.

### select_action

This function selects an action based on an epsilon-greedy policy, balancing exploration and exploitation.

### train

This function performs the training process of the neural network using random samples from the replay memory.
