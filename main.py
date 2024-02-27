#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:21:46 2022

@author: Gianmarco Lavacca, Giordano Luchi
"""


'''
 ___                                 _        
|_ _| _ __ ___   _ __    ___   _ __ | |_  ___ 
 | | | '_ ` _ \ | '_ \  / _ \ | '__|| __|/ __|
 | | | | | | | || |_) || (_) || |   | |_ \__ \
|___||_| |_| |_|| .__/  \___/ |_|    \__||___/
                |_|                           
'''
import os
import time
import random
from tqdm import trange
from itertools import product
import art
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow.keras.optimizers as opt
from pendulum import Pendulum

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()




'''
 _   _                                                                        _                    
| | | | _   _  _ __    ___  _ __  _ __    __ _  _ __   __ _  _ __ ___    ___ | |_   ___  _ __  ___ 
| |_| || | | || '_ \  / _ \| '__|| '_ \  / _` || '__| / _` || '_ ` _ \  / _ \| __| / _ \| '__|/ __|
|  _  || |_| || |_) ||  __/| |   | |_) || (_| || |   | (_| || | | | | ||  __/| |_ |  __/| |   \__ \
|_| |_| \__, || .__/  \___||_|   | .__/  \__,_||_|    \__,_||_| |_| |_| \___| \__| \___||_|   |___/
        |___/ |_|                |_|                                                               
'''
# Model Settings
DOFS                            = 1
ACTIONS                         = 7
ARMS_LENGTH                     = [1.] if DOFS==1 else [1., 1.]
TRAINING_TOLERANCE              = 1e-3
TESTING_TOLERANCE               = 0.2
POSSIBLE_COMBINATIONS           = np.power(ACTIONS, DOFS)
RENDER_TRAINING                 = False
RENDER_TESTING                  = True
LOAD_DEMO                       = False
EXPERIMENT                      = '0'
TEST                            = 1
MODEL_NAME                      = "Experiment_{}_Actions_{}".format(EXPERIMENT, ACTIONS)
DEMO_NAME                       = "DEMO_{}".format(ACTIONS)

# Episodes Settings
TRAINING_EPISODES               = 2_000
TESTING_EPISODES                = 10
EPISODE_LENGTH                  = 50*DOFS
FORCE_STOP                      = False

# Memory Settings
REPLAY_MEMORY_SIZE              = int(TRAINING_EPISODES*EPISODE_LENGTH/5)
MAX_MEMORY_EPISODES             = 50
BATCH_SIZE                      = 64
MIN_FULL_MEMORY                 = 100*BATCH_SIZE

#Learning Settings
Q_WEIGHT                        = 5 if DOFS==1 else [5, 5]
V_WEIGHT                        = 0.01
U_WEIGHT                        = 0.3
Q_LEARNING_RATE                 = 5e-4
DISCOUNT                        = 0.9
UPDATE_FREQ                     = int(60*EPISODE_LENGTH/100)
MOVING_AVERAGE_SIZE             = 100

# Exploration Settings
EPSILON                         = 1.0
EPSILON_MIN                     = 0.001
FULL_EXPLOITATION_AT_EPISODE    = int(90*TRAINING_EPISODES/100)
EPSILON_DECAY                   = np.exp(np.log(EPSILON_MIN)/FULL_EXPLOITATION_AT_EPISODE)
   # Equally, also              = np.power(EPSILON_MIN, (1/FULL_EXPLOITATION_AT_EPISODE))

# Plot Settings
PLOT_FREQ                       = int(TRAINING_EPISODES/10)




'''
 ____   _         _     _____                      _    _                    
|  _ \ | |  ___  | |_  |  ___| _   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
| |_) || | / _ \ | __| | |_   | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
|  __/ | || (_) || |_  |  _|  | |_| || | | || (__ | |_ | || (_) || | | |\__ \
|_|    |_| \___/  \__| |_|     \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
'''
class Plots:
    def __init__(self, env, path, n_fig=1):
        ''' Initializes the plotter that keeps track of the number of
            figures already created and the path to which they are saved.
        '''
        self.path           = path
        self.env            = env
        self.n_fig          = n_fig
        self.torque_step    = 4/env.dsu
        self.umax           = env.umax
        self.vmax           = env.vmax
    
    def set_nfig(self, n_fig=1):
        ''' Sets the n_fig attribute to the passed value. '''
        self.n_fig = n_fig
    
    def plot_decay_rate(self):
        ''' Plots the decay rate function that dictates the values
            of Epsilon throughout the training.
            Saves the plot to a '.png' image.
        '''
        x = np.arange(0, TRAINING_EPISODES, 1)
        # Decay rate goes like exp(-x)
        y = np.exp(-(1 - EPSILON_DECAY) * x)
        
        mark = TRAINING_EPISODES - 1
        for i in range(len(x)):
            if y[i]<=0.001:
                mark = i
                plt.scatter(mark, y[mark], s=75, c='red', zorder=3,
                            label='< 0.001 at episode {}'.format(mark))
                break
            elif i==len(x)-1:
                plt.scatter(mark, y[mark], label='never < 0.001',
                            marker='X', s=75, c='red', zorder=2)
        plt.plot(x, y, label='Decay Function', color='black', zorder=2)
        plt.scatter(0.5, 0.5, s=0, zorder=1,
                    label='Last Epsilon is {}'.format(np.round(y[-1], 7)))
        plt.ylabel('Epsilon')
        plt.xlabel('Episodes')
        plt.grid(b=True, which='both', ls='--')
        plt.legend()
        plt.title('Epsilon Decay Rate')
        
        plt.savefig(get_path(self.path, 'Epsilon_Decay_Rate.png'),
                    bbox_inches='tight')
        plt.show()
    
    def plot_training(self, x_range, cost_hist, avg_cost_hist):
        ''' Plots the cost history for each episode and the average
            cost over 50 past episodes. Also highlights the best (minimum)
            average cost overall.
            Saves the plot to a '.png' image.
        '''
        x = np.arange(0, x_range + 1, 1)
        
        plt.figure(self.n_fig)
        # Plots cost history
        plt.plot(x, tf2np(cost_hist), color='black', label='Episodic',
                 linewidth=2, zorder=1)
        # Plots average cost history
        plt.plot(x, avg_cost_hist, color='red', label='Average',
                 linewidth=2, zorder=2)
        if len(avg_cost_hist)>200:
            # Plots best average cost on x axis
            plt.vlines(np.argmin(avg_cost_hist[200:])+200, ymin=0, ymax=min(avg_cost_hist[200:]),
                       colors='yellow', linestyles='dotted', linewidth=3, zorder=3)
            # Plots best average cost on y axis
            plt.hlines(min(avg_cost_hist[200:]), xmin=0, xmax=np.argmin(avg_cost_hist[200:])+200,
                       colors='yellow', linestyles='dotted', linewidth=3, zorder=4)
            plt.scatter(np.argmin(avg_cost_hist[200:])+200, min(avg_cost_hist[200:]),
                        label='Best Average: {}'.format(np.round(min(avg_cost_hist[200:]), 1)),
                        s=75, c='yellow', zorder=5)
        
        plt.ylabel("Cost")
        plt.xlabel("Training Episodes")
        plt.legend(loc='upper right')
        plt.title("Training Cost Behaviour")
        
        # Saves every checkpoint plot on a '.png' file
        if (x_range + 1)==TRAINING_EPISODES:
            plt.savefig(get_path(self.path,
                                 ('Training_Cost_Behaviour_Final.png'
                                  if not LOAD_DEMO else
                                  'DEMO_Training_Cost_Behaviour_Final.png')),
                        bbox_inches='tight')
        else:
            plt.savefig(get_path(self.path,
                                 ('Training_Cost_Behaviour_{}.png'.format(self.n_fig)
                                 if not LOAD_DEMO else
                                 'DEMO_Training_Cost_Behaviour_{}.png'.format(self.n_fig))),
                        bbox_inches='tight')
        plt.show()
        
        self.n_fig += 1
    
    def plot_testing(self, state, action, cost, step):
        ''' Plots the angular position and the angular velocity
            of each DOF during the episode. Also plots the torque
            applied to the joint.
            Saves the plot to a '.png' image.
        '''
        for i in range(len(state)):
            state[i,0] = self.env.to_AbsRefSys(state[i,0])
        
        x = np.arange(0, step, 1)
        colors = ['green', 'purple', 'black', 'red', 'blue', 'cyan']
        if DOFS>6:
            print("Not enough colors!")
            return
        
        plt.figure(self.n_fig)
        
        plt.subplot(311)
        plt.scatter(0, -np.pi, s=0, c='white',
                    label='Episode Cost: {}'.format(np.round(cost, 1)))
        # Plots the behaviour of the position during the test
        for i in range(DOFS):
            if DOFS==1:
                plt.plot(x, state[:,0], color='green', linewidth=2)
            else:
                plt.plot(x, state[:,0,i], color=colors[i], linewidth=2,
                         label='DOF {}'.format(i+1))
        plt.ylabel("Angular Position")
        theta = np.arange(-np.pi, np.pi+np.pi/2, np.pi/2)
        plt.yticks(theta, ['-π', '-π/2', '0', 'π/2', 'π'])
        plt.grid(b=True, which='both', axis='y', ls='--')
        plt.title("Test {} - Episode {}".format(TEST, self.n_fig))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5,-2.85), ncol=3)
        
        plt.subplot(312)
        # Plots the behaviour of the velocity during the test
        for i in range(DOFS):
            if DOFS==1:
                plt.plot(x, state[:,1], color='purple', linewidth=2)
            else:
                plt.plot(x, state[:,1,i], color=colors[i], linewidth=2)
        speed = np.arange(-self.vmax, self.vmax + 4, 4)
        plt.yticks(speed)
        plt.ylabel("Velocity")
        plt.grid(b=True, which='both', axis='y', ls='--')
        
        plt.subplot(313)
        # Plots the behaviour of the applied torque during the test
        for i in range(DOFS):
            if DOFS==1:
                plt.plot(x, action, color='black', linewidth=2)
            else:
                plt.plot(x, action[:,i], color=colors[i], linewidth=2)
        torque = np.arange(-self.umax, self.umax + 2*self.torque_step,
                           2*self.torque_step)
        plt.yticks(torque)
        plt.grid(b=True, which='both', axis='y', ls='--')
        plt.ylabel("Torque")
        plt.xlabel("Episode Length [steps]")
        
        plt.savefig(get_path(self.path,
                             ("Test_{}_Episode_{}.png".format(TEST, self.n_fig)
                             if not LOAD_DEMO else
                             "DEMO_{}_Episode_{}.png".format(ACTIONS, self.n_fig))),
                    bbox_inches='tight')
        plt.show()
        
        self.n_fig += 1
    
    def log_hyperparams(self):
        ''' Logs and formats the hyperparameters of the current
            experiment in a '.txt' file.
        '''
        hyperparams = \
        ["Model Settings",
         ["DOFS", DOFS],
         ["ACTIONS", ACTIONS],
         ["ARMS_LENGTH", ARMS_LENGTH],
         ["TRAINING_TOLERANCE", TRAINING_TOLERANCE],
         ["TESTING_TOLERANCE", TESTING_TOLERANCE],
         ["POSSIBLE_COMBINATIONS", POSSIBLE_COMBINATIONS],
         ["RENDER_TRAINING", RENDER_TRAINING],
         ["RENDER_TESTING", RENDER_TESTING],
         ["LOAD_DEMO", LOAD_DEMO],
         ["MODEL_NAME", MODEL_NAME] if not LOAD_DEMO else ["DEMO_NAME", DEMO_NAME],
         "Episodes Settings",
         ["TRAINING_EPISODES", TRAINING_EPISODES],
         ["TESTING_EPISODES", TESTING_EPISODES],
         ["EPISODE_LENGTH", EPISODE_LENGTH],
         ["FORCE_STOP", FORCE_STOP],
         "Memory Settings",
         ["REPLAY_MEMORY_SIZE", REPLAY_MEMORY_SIZE],
         ["MAX_MEMORY_EPISODES", MAX_MEMORY_EPISODES],
         ["BATCH_SIZE", BATCH_SIZE],
         ["MIN_FULL_MEMORY", MIN_FULL_MEMORY],
         "Learning Settings",
         ["Q_WEIGHT", Q_WEIGHT],
         ["V_WEIGHT", V_WEIGHT],
         ["U_WEIGHT", U_WEIGHT],
         ["Q_LEARNING_RATE", Q_LEARNING_RATE],
         ["DISCOUNT", DISCOUNT],
         ["UPDATE_FREQ", UPDATE_FREQ],
         ["MOVING_AVERAGE_SIZE", MOVING_AVERAGE_SIZE],
         "Exploration Settings",
         ["EPSILON", EPSILON],
         ["EPSILON_MIN", EPSILON_MIN],
         ["FULL_EXPLOITATION_AT_EPISODE", FULL_EXPLOITATION_AT_EPISODE],
         ["EPSILON_DECAY", EPSILON_DECAY],
         "Plot Settings",
         ["PLOT_FREQ", PLOT_FREQ]]
        
        # Overwrites any existing log file with the same name
        if not LOAD_DEMO:
            log_file = open(get_path(self.path,
                                     "Hyperparams_{}_DOFs_Experiment_{}_Test_{}.txt"
                                     .format(DOFS, EXPERIMENT, TEST)),
                            "w")
        else:
            log_file = open(get_path(self.path,
                                     "Hyperparams_DEMO_{}_DOFs.txt"
                                     .format(DOFS)),
                            "w")
        
        log_file.write("HYPERPARAMETERS:\n")
        # Formats the log file
        for i in range(len(hyperparams)):
            if len(hyperparams[i])!=2:
                log_file.write("\n{}:\n".format(hyperparams[i]))
            else:
                log_file.write("{}:".format(hyperparams[i][0])
                               + "."*(34 - len(hyperparams[i][0]))
                               + "{}\n".format(hyperparams[i][1]))
        
        log_file.close()
    
    def log_goal(self, init_state, cost, episode, steps):
        if LOAD_DEMO:
            return
        
        log_file = open(get_path(self.path,
                                 "Goals_{}_DOFs_Experiment_{}.txt"
                                 .format(DOFS, EXPERIMENT)),
                        "a")
        
        log_file.write("GOAL!\n")
        log_file.write("   ⎡Episode: {}\n".format(episode))
        log_file.write("   ⎢Total Steps: {}\n".format(steps))
        log_file.write("   ⎢Episode Cost: {}\n".format(cost))
        # Single DOF case
        if DOFS==1:
            log_file.write("   ⎢Initial State: ⎡θ = {}\n".format(np.round(init_state[0], 4)))
            log_file.write("   ⎣" + "."*15 +  "⎣θ_dot = {}\n".format(np.round(init_state[1], 4)))
        # Multiple DOFs case
        elif DOFS>=2:
            position = "   ⎢Initial State: ⎡("
            for i in range(DOFS):
                if i==DOFS-1:
                    position = position + "θ{}".format(i+1)
                else:
                    position = position + "θ{}, ".format(i+1)
            position = position + ") = ("
            for i in range(DOFS):
                if i==DOFS-1:
                    position = position + "{}".format(np.round(init_state[0,i], 4))
                else:
                    position = position + "{}, ".format(np.round(init_state[0,i], 4))
            position = position + ")\n"
            
            velocity = "   ⎣" + "."*15 +  "⎣("
            for i in range(DOFS):
                if i==DOFS-1:
                    velocity = velocity + "θ{}_dot".format(i+1)
                else:
                    velocity = velocity + "θ{}_dot, ".format(i+1)
            velocity = velocity + ") = ("
            for i in range(DOFS):
                if i==DOFS-1:
                    velocity = velocity + "{}".format(np.round(init_state[1,i], 4))
                else:
                    velocity = velocity + "{}, ".format(np.round(init_state[1,i], 4))
            velocity = velocity + ")\n\n"
            
            log_file.write(position)
            log_file.write(velocity)
        
        log_file.close()




'''
 _   _  _    _  _  _  _            _____                      _    _                    
| | | || |_ (_)| |(_)| |_  _   _  |  ___| _   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
| | | || __|| || || || __|| | | | | |_   | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
| |_| || |_ | || || || |_ | |_| | |  _|  | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 \___/  \__||_||_||_| \__| \__, | |_|     \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
                           |___/                                                        
'''
def set_path():
    ''' Returns the path of the directory found or created in the
        current working folder. This will be the directory where
        all images and data will be saved.
    '''
    cwd = os.getcwd()
    if not LOAD_DEMO:
        final_path = os.path.join(cwd, '{}_DOFs_Experiment_{}'
                                       .format(DOFS, EXPERIMENT))
    else:
        final_path = os.path.join(cwd, 'DEMO_{}_DOFs'.format(DOFS))
    
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    return final_path

def get_path(path, filename):
    ''' Returns the joined path where the file will be saved,
        including the filename.
    '''
    return os.path.join(path, filename)

def clean_folder(path):
    ''' Removes the partial cost behaviour plots from the
        experiment folder.
    '''
    for i in range(9):
        rm_path = get_path(path,
                           ('Training_Cost_Behaviour_{}.png'.format(i+1)
                           if not LOAD_DEMO else
                           'DEMO_Training_Cost_Behaviour_{}.png'.format(i+1)))
        os.remove(rm_path)

def set_seed(env, seed_value):
    ''' Sets random seeds '''
    random_seed = int((time.time()%10)*1000)
    print("Seed = {}".format(random_seed))
    np.random.seed(random_seed)

def np2tf(y, axis, transposed=True):
    ''' Converts from numpy to tensorflow along a desired axis.
        Allows for not transposing the output through a flag.
    '''
    if transposed:
        out = tf.expand_dims(tf.convert_to_tensor(y), axis).T
    else:
        out = tf.expand_dims(tf.convert_to_tensor(y), axis)
    return out
    
def tf2np(y):
    ''' Converts from tensorflow to numpy '''
    return tf.squeeze(y).numpy()

def map_action(env, action_idx=None, action=None, get_index=False):
    ''' Either maps the passed index to the corresponding action 
        or maps the passed action to the corresponding index.
    '''
    if get_index==False:
        if action_idx >= POSSIBLE_COMBINATIONS:
            print("\n\n\nIndex is out of bounds\n\n\n")
            return None
        # Returns the action corresponding to the passed index:
        # From index [0] to [env.n_combos] returns an action with torque
        # from value -2  to      +2        respectively with 0.2 step
        if DOFS==1:
            return np.round_(env.action_space[action_idx], 1)
        elif DOFS>=2:
            combo = list(product(env.action_space.tolist(), repeat=DOFS))
            action_combo = combo[action_idx]
            return np.round_(action_combo, 1)
    else:
        if action.any()==None:
            print("\n\n\nNo action provided\n\n\n")
            return None
        # Returns the index corresponding to the passed action:
        # From action between -2  to      +2        with 0.2 step returns
        # an index from       [0] to [env.n_combos] respectively
        if DOFS==1:
            return np.where(np.round(env.action_space, 1)
                            == float(action))[0][0]
        elif DOFS>=2:
            combo = list(product(range(len(env.action_space)), repeat=DOFS))
            index_combo = []
            for i in range(env.nu):
                index_combo.append(np.where(np.round(env.action_space, 1)
                                            == float(action[i]))[0][0])
            return combo.index(tuple(index_combo))




'''
 ____                _                 __  __                                      
|  _ \   ___  _ __  | |  __ _  _   _  |  \/  |  ___  _ __ ___    ___   _ __  _   _ 
| |_) | / _ \| '_ \ | | / _` || | | | | |\/| | / _ \| '_ ` _ \  / _ \ | '__|| | | |
|  _ < |  __/| |_) || || (_| || |_| | | |  | ||  __/| | | | | || (_) || |   | |_| |
|_| \_\ \___|| .__/ |_| \__,_| \__, | |_|  |_| \___||_| |_| |_| \___/ |_|    \__, |
             |_|               |___/                                         |___/ 
'''
class ReplayMemory:
    def __init__(self, env):
        ''' Initializes the replay memory with given capacity.
            Allocates arrays of the apprpriate shape for:
                States          (REPLAY_MEMORY_SIZE, 2, 1)
                Actions         (REPLAY_MEMORY_SIZE, 1, 1)
                Costs           (REPLAY_MEMORY_SIZE,     )
                Next states     (REPLAY_MEMORY_SIZE, 2, 1)
                Dones           (REPLAY_MEMORY_SIZE,     )
            Initializes an index for keeping track of the position to 
            which the arrays have been filled.
        '''
        self.capacity       = REPLAY_MEMORY_SIZE
        self.max_memory     = self.capacity
        self.states         = np.zeros((self.capacity, *(env.nx, 1)))
        self.actions        = np.zeros((self.capacity, *(env.nu, 1)))
        self.costs          = np.zeros(self.capacity)
        self.next_states    = np.zeros((self.capacity, *(env.nx, 1)))
        self.dones          = np.zeros(self.capacity)
        
        self.index = 0
        self.env = env
        
    def store(self, state, action, cost, next_state, done):
        ''' Sets the value of the next-to-last filled position of the
            corresponding array to the value passed to the method.
            Starts again from the first position after the arrays are
            full.
        '''
        index = self.index % self.capacity
        
        self.states[index]      = np.reshape(state, (self.env.nx,1))
        self.actions[index]     = np.reshape(action, (self.env.nu,1))
        self.costs[index]       = cost
        self.next_states[index] = np.reshape(next_state, (self.env.nx,1))
        self.dones[index]       = int(done)
        
        self.index += 1
        
    def sample(self):
        ''' Uses a randomly generated batch of indexes to extract the 
            values of each array at those positions.
            Returns arrays containing the extracted values.
        '''
        self.max_memory = min(self.index, self.capacity)
        batch = random.sample(range(self.max_memory), BATCH_SIZE)
        
        states      = self.states[batch]
        actions     = self.actions[batch]
        costs       = self.costs[batch]
        next_states = self.next_states[batch]
        dones       = self.dones[batch]
        
        return states, actions, costs, next_states, dones
    
    def fill_memory(self, agent):
        ''' Fills the arrays of the Replay Memory with random states and
            actions (and what next_states and costs they produce) for
            the agent to have something to sample in the beginning of the
            learning part.
        '''
        art.tprint("Filling\nMemory", font="starwars")
        
        for episode in trange(MAX_MEMORY_EPISODES,
                      ncols=100, desc="Episodes", unit=' Episodes',
                      colour='green'):
            state = self.env.reset()
            
            for step in trange(int(REPLAY_MEMORY_SIZE/MAX_MEMORY_EPISODES),
                               ncols=100, desc="Episode {}".format(episode),
                               miniters=1000, delay=1e-7, unit=' Steps',
                               colour='red'):
                action = agent.choose_action(state)
                next_state, cost, done = self.env.step(action)
                self.store(state, action, cost, next_state, done)
                state = next_state
    
    def __len__(self):
        ''' Returns the number of cells already filled '''
        return self.max_memory
    
#################################### REPLAY MEMORY TEST
#env = Pendulum()
#mem = ReplayMemory(env)
#print(mem.costs.shape, "\n", mem.costs)




'''
 __  __             _        _ 
|  \/  |  ___    __| |  ___ | |
| |\/| | / _ \  / _` | / _ \| |
| |  | || (_) || (_| ||  __/| |
|_|  |_| \___/  \__,_| \___||_|
'''
def build_DQN(env, model_name):
    ''' Generates the layers of a Neural Network with the shape of the 
        inputs and the outputs inferred from the observation space and 
        action shape of the environment passed.
        Groups those layers into an object with training and inference
        features. The Model is instantiated with the "Functional API".
        Prints a summary of the NN.
    '''
    input_dims = [env.nx, env.n_combos]
    
    inputs      = layers.Input(shape=(1, input_dims[0]))
    state_out1  = layers.Dense(16, activation="relu")(inputs) 
    state_out2  = layers.Dense(32, activation="relu")(state_out1) 
    state_out3  = layers.Dense(64, activation="relu")(state_out2) 
    state_out4  = layers.Dense(64, activation="relu")(state_out3)
    outputs     = layers.Dense(input_dims[1])(state_out4)

    model = Model(inputs, outputs, name=model_name)
    model.compile(optimizer=opt.RMSprop(Q_LEARNING_RATE), loss="mse",
                  metrics=['accuracy'])
    
    model.summary()

    return model

############################################ MODEL TEST
#env = Pendulum()
#Q_Network = build_DQN(env, MODEL_NAME)
#Q_Target_Network = build_DQN(env, MODEL_NAME)

########################### WEIGHTS CHECK
#w = Q_Network.get_weights()
#print("\n\n Weights Check \n")
#for i in range(len(w)):
#    print("Q_Network weights layer", i, np.linalg.norm(w[i]))
#
#print("\nDouble the weights")
#for i in range(len(w)):
#    w[i] *= 2
#Q_Network.set_weights(w)
#
#w = Q_Network.get_weights()
#for i in range(len(w)):
#    print("Q_Network weights layer", i, np.linalg.norm(w[i]))
#
#print("\nSave NN weights to file (in HDF5)")
#Q_Network.save_weights("Weights_Check.h5")
#
#print("Load NN weights from file\n")
#Q_Target_Network.load_weights("Weights_Check.h5")
#
#w = Q_Target_Network.get_weights()
#for i in range(len(w)):
#    print("Q_Network weights layer", i, np.linalg.norm(w[i]))
#print("\n\n")




'''
    _                         _   
   / \     __ _   ___  _ __  | |_ 
  / _ \   / _` | / _ \| '_ \ | __|
 / ___ \ | (_| ||  __/| | | || |_ 
/_/   \_\ \__, | \___||_| |_| \__|
          |___/                   
'''
class DQN_Agent:
    def __init__(self, env, plotter, path, train_mode):
        ''' Initializes the Agent that will manage the Replay Memory
            and the NNs and their learning.
        '''
        # Utility
        self.path = path
        self.env = env
        self.action_space = env.action_space
        self.train_mode = train_mode
        
        # Initialize epsilon settings
        self.eps        = EPSILON
        self.eps_min    = EPSILON_MIN
        self.eps_decay  = EPSILON_DECAY
        plotter.plot_decay_rate()
        
        # Initialize Replay Memory
        self.replay_memory = ReplayMemory(env)
        
        # Create main model (Will be trained every step)
        self.Q = build_DQN(env, "Q")
        
        # Create target model (Will predict the Q values every step)
        self.Q_Target = build_DQN(env, "Q_Target")
        # Set the weights of the target model equal to the weights
        # of the main model
        self.Q_Target.set_weights(self.Q.get_weights())
        
        # Instantiate an optimizer
        self.model_optimizer = opt.RMSprop(Q_LEARNING_RATE)
    
    def update_target(self):
        ''' Set the weights of the target model equal to the weights
            of the main model.
        '''
        self.Q_Target.set_weights(self.Q.get_weights())
        
    def choose_action(self, state):
        ''' Chooses between performing a random action and performing
            the best action, as predicted by the target model, with 
            probability given by the value of "self.epsilon"
            The higher epsilon the more random the actions.
        '''
        if random.uniform(0,1) > self.eps:
            state = state.reshape((1, 1, self.env.nx))
            Q_value = abs(self.Q(state, training=self.train_mode))
            min_Q_idx = np.unravel_index(np.argmin(Q_value), Q_value.shape)[2]
            action = map_action(self.env, action_idx=min_Q_idx)
            return np2tf(np.round_(action, 1), 0)
        else:
            return np2tf(np.round_(np.random.choice(env.action_space, DOFS), 1), 0)
        
    def update_epsilon(self):
        ''' Updates the value of epsilon to the highest value between
            the minimum wanted epsilon value and the current value
            multiplied by a given decay rate.
        '''
        self.eps = max(self.eps_min, self.eps*self.eps_decay)
        
    def reset_epsilons_to_zero(self):
        ''' Sets all epsilon parameters to zero for the testing part '''
        self.eps        = 0.0
        self.eps_min    = 0.0
        self.eps_decay  = 0.0
        
    def get_train_mode(self, train_mode):
        ''' Sets the train_mode flag to distinguish between training
            and testing
        '''
        self.train_mode = train_mode
        
    def learn(self, done):
        ''' Tunes the NNs based on previous steps '''
        # Starts learning only if there already is a specified amount of
        # samples in the Replay Memory
        if self.replay_memory.__len__() < MIN_FULL_MEMORY:
            return
        
        # Sample and process a batch of data from the Replay Memory
        states,actions,costs,next_states,dones = self.replay_memory.sample()
        states = tf.reshape(states, (BATCH_SIZE, 1, self.env.nx))
        next_states = tf.reshape(next_states, (BATCH_SIZE, 1, self.env.nx))
        actions = np.reshape(np.round_(actions, 1), [BATCH_SIZE,DOFS])
        actions_idxs = []
        for i in range(BATCH_SIZE):
            actions_idxs.append(map_action(self.env, action=actions[i],
                                           get_index=True))
        costs = tf.reshape(costs, (BATCH_SIZE,1,1))
        
        with tf.GradientTape() as tape:
            if not done:
                # Compute the target Q values with the target model
                Q_Target_values = abs(self.Q_Target(next_states,
                                                    training=self.train_mode))
                min_Q_Tar_values = tf.math.reduce_min(Q_Target_values,
                                                      axis=2, keepdims=True)
                #Compute 1-step targets for the critic loss
                y = costs + DISCOUNT*min_Q_Tar_values
            else:
                #Compute 1-step targets for the critic loss
                y = costs
            
            # Compute the Q values with the main model
            Q_values = abs(self.Q(states, training=self.train_mode))
            taken_Q_values = tf.gather(Q_values, actions_idxs,
                                       axis=-1, batch_dims=1)
            taken_Q_values = taken_Q_values.reshape(tf.shape(y))
            # Compute loss function
            Q_Loss = tf.math.reduce_mean(tf.math.square(y - taken_Q_values))
        
        # Compute the gradients of the critic loss w.r.t. critic's parameters
        # (weights and biases)
        Q_gradient = tape.gradient(Q_Loss, self.Q.trainable_variables)
        # Update the critic backpropagating the gradients
        self.model_optimizer.apply_gradients(zip(Q_gradient,
                                                 self.Q.trainable_variables))
        
        # Recompile both model with the updated loss function
        self.Q.compile(optimizer=self.model_optimizer, loss=Q_Loss)
        self.Q_Target.compile(optimizer=self.model_optimizer, loss=Q_Loss)
        
    def save(self):
        ''' Saves the weights of the main model to a .h5 file '''
        self.Q.save_weights(get_path(self.path, '{}.h5'.format(MODEL_NAME)),
                            overwrite=True)
        
    def load(self):
        ''' Loads the weights of the main model from a .h5 file '''
        if not LOAD_DEMO:
            model_path = get_path(self.path, '{}.h5'.format(MODEL_NAME))
            self.Q.load_weights(model_path)
        else:
            demo_path = get_path(self.path, '{}.h5'.format(DEMO_NAME))
            self.Q.load_weights(demo_path)
    



'''
 __  __         _        
|  \/  |  __ _ (_) _ __  
| |\/| | / _` || || '_ \ 
| |  | || (_| || || | | |
|_|  |_| \__,_||_||_| |_|
'''
def train(env, agent, plotter):
    ''' Trains the model to reach the desired goal '''
    if LOAD_DEMO or TEST>1:
        return False
    
    # Fill the Replay Memory
    agent.replay_memory.fill_memory(agent)
    
    # Initialize step counter, cost history array and best score
    step_cnt    = 0
    best_score  = np.inf
    cost_hist = []
    avg_cost_hist = []
    
    art.tprint("Training", font="starwars")
    
    # Loop through Training Episodes
    for ep_cnt in trange(TRAINING_EPISODES,
                         ncols=100, desc="Training", unit='Episodes',
                         delay=0.1, colour='green'):
        # Reset the state to a random value
        state = env.reset()
        init_state = state
        # Initialize done flag and episode cost
        done        = False
        ep_cost   = 0
        
        # The Episode goes on until either the goal is reached
        # or the maximum steps allowed for episode are hit
        for step in trange(EPISODE_LENGTH,
                           ncols=100, desc="Episode {}".format(ep_cnt),
                           mininterval=(9 if RENDER_TRAINING else 5),
                           unit=' Steps', colour='red'):
            # Chooses action based on the value of Epsilon
            action = agent.choose_action(state)
            # Performs a simulation step with the chosen action
            next_state, cost, done = env.step(action)
            
            # Updates the Replay Memory with the data from this step
            agent.replay_memory.store(state, action, cost, next_state, done)
            # Updates the NNs with the updated Replay Memory
            agent.learn(done)
            
            # Update the weights of the target model to match those
            # of the main model every UPDATE_FREQ steps
            if (step_cnt % UPDATE_FREQ == 0):
                agent.update_target()
            
            # Update the cost
            ep_cost += cost
            # Propagate the state
            state = next_state
            # Update the counter
            step_cnt += 1
            # Render the step
            if RENDER_TRAINING:
                env.render()
            
            # If goal has been reached, log it and finish the episode
            if done:
                plotter.log_goal(init_state, tf2np(ep_cost), ep_cnt, step_cnt%EPISODE_LENGTH)
                art.tprint("\n      Goal Reached!", font="small")
                break
        
        # Update the Epsilon every Episode
        past_eps = agent.eps
        agent.update_epsilon()
        # Update the cost history array
        cost_hist.append(ep_cost)
        # Calculate moving average over last 100 episodes
        curr_avg_score = np.mean(cost_hist[-MOVING_AVERAGE_SIZE:])
        avg_cost_hist.append(curr_avg_score)
        
        print('''
                 ###  Ep: {}, Total Steps: {}
                 ###  Ep Score {}, Current Average Score: {}
                 ###  Epsilon: {}, Next Epsilon: {}
              '''
              .format(ep_cnt, step_cnt, tf2np(ep_cost),
                      curr_avg_score, past_eps, agent.eps))
        
        # Saves the weights of the main model of this episode if the resulting
        # average score is lower than the last best average score and updates
        # the best average score
        if (ep_cnt > 200) and (done or (curr_avg_score <= best_score)):
            agent.save()
            best_score = curr_avg_score
        
        # Plots the costs of past episodes every PLOT_FREQ
        if ((ep_cnt + 1) % PLOT_FREQ == 0):
            plotter.plot_training(ep_cnt, cost_hist, avg_cost_hist)
    # Cleans output folder
    clean_folder(agent.path)
    
    return False

def test(env, agent, plotter):
    ''' Tests the model on what it learned in the Training'''
    
    if LOAD_DEMO:
        art.tprint("DEMO", font="starwars")
    else:
        art.tprint("Testing", font="starwars")
    
    # Resets the figure number of the plotter after the training
    plotter.set_nfig()
    
    # Loop through Testing Episodes
    for ep_cnt in trange(TESTING_EPISODES,
                        ncols=100, desc="Testing", unit=' Episodes',
                        colour='green'):
        # Reset the state to a random value
        state = env.reset()
        # Initialize done flag and episode cost
        done        = False
        ep_cost   = 0
        state_hist = []
        action_hist = []
        # Initialize counter to keep track of the episode length
        step = 0
        
        # The Episode goes on until the pendulum reaches the goal
        while not done:
            # Chooses action: Epsilon is 0 so the action is not random
            action = agent.choose_action(state)
            action_hist.append(tf2np(action))
            # Perform a simulation step with the chosen action
            next_state, cost, done = env.step(action)
            # Propagate the state
            state_hist.append(tf2np(state))
            state = next_state
            # Update the cost
            ep_cost += cost
            # Update the length counter
            step += 1
            # Render the step
            if RENDER_TESTING:
                env.render()
            
            # Forces stop at 100 steps if the flag is true
            if FORCE_STOP and step==100:
                done = True
        
        # Plots the episode's data
        plotter.plot_testing(np.asarray(state_hist), np.asarray(action_hist),
                             tf2np(ep_cost), step)
        
#        input("Press Enter to continue...")
    
    return True

if __name__ == '__main__':
    path = set_path()
    # Environment initialization
    weights = [Q_WEIGHT, V_WEIGHT, U_WEIGHT]
    env = Pendulum(ACTIONS, ARMS_LENGTH, weights, DOFS)
    env.set_Tolerance(TRAINING_TOLERANCE)
    # Plotter initialization
    plotter = Plots(env, path)
    # Hyperparameters logging
    plotter.log_hyperparams()
    # Logic to perform training and testing phases
    train_mode  = True
    tested      = False
    
    while tested==False:
        if train_mode:
            # Training seed - Different from Testing seed
            set_seed(env, 0)
            # Agent initialization
            agent = DQN_Agent(env, plotter, path, train_mode)
            # Training
            train_mode = train(env, agent, plotter)
        else:
            # Set new tolerance
            env.set_Tolerance(TESTING_TOLERANCE)
            # Testing seed - Different from Training seed
            set_seed(env, 10)
            # Agent updates its train_mode flag
            agent.get_train_mode(train_mode)
            # Agent resets its epsilons values
            agent.reset_epsilons_to_zero()
            # Agent loads the model saved during training
            agent.load()
            # Testing
            tested = test(env, agent, plotter)
