import numpy as np
from collections import deque
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from utility import state_array, save_data



class SARSA:
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes):
        self.env=env
        self.memory = deque(maxlen=2048)  # Deque is quick for append and pop operations from both the ends of the container.
        self.alpha=alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberEpisodes = numberEpisodes
        self.numberActions= self.env.action_space.n
        self.numberStates= self.env.observation_space.n
        self.reward_cache = list()
        self.observation_cache = list()
        self.step_cache = list()
        self.action_cache=list()
        self.episode=0

        #state value
        self.learnedPolicy= np.zeros(self.env.observation_space.n)

        #Action value table

        self.Qtable= np.zeros((self.env.observation_space.n,self.env.action_space.n))

    # Add each step of an episode to agent's memory
    def add_memory(self, state, action, reward, new_state, terminated):
        # Insert the value in its argument to the right end of the deque
        self.memory.append((state, action, reward, new_state, terminated))

    def selectAction(self, state):
        #epsilon greedy policy

        if np.random.random()<self.epsilon:
            return np.random.choice(self.numberActions)

        else:
            return np.argmax(self.Qtable[state,:])

    def simulateEpisodes(self):

        # here we loop through the episodes
        for episode in range(self.numberEpisodes):

            # reset the environment at the beginning of every episode
            (currState, prob) = self.env.reset()
            # self.observation_cache.append(currState)
            state_arr = state_array(self.numberStates, currState)
            print("resetting environment for episode ", episode ," with initial state: " , currState," :" , prob)

            # select an action on the basis of the initial state
            currAction = self.selectAction(currState)

            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            while not terminalState:

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (nextState, reward, terminalState, _, _) = self.env.step(currAction)
                new_state_arr = state_array(self.numberStates, nextState)
                # self.observation_cache.append(nextState)
                # next action
                nextAction = self.selectAction(nextState)

                if not terminalState:
                    error = reward + self.gamma * self.Qtable[nextState, nextAction] - self.Qtable[currState, currAction]
                    self.Qtable[currState, currAction] = self.Qtable[currState, currAction] + self.alpha * error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0
                    error = reward - self.Qtable[currState, currAction]
                    self.Qtable[currState, currAction] = self.Qtable[currState, currAction] + self.alpha * error
                self.reward_cache.append(reward)
                # Add current step to agent's memory
                self.add_memory(state_arr.tolist(), currAction, reward, new_state_arr.tolist(),terminalState)
                currState = nextState
                currAction = nextAction
        save_data(self.memory,"SARSA")

    # now we compute the final learned policy
    def computeFinalPolicy(self):
            for indexS in range(self.numberStates):
                # we use np.random.choice() because in theory, we might have several identical maximums
                self.learnedPolicy[indexS] = np.random.choice(
                    np.where(self.Qtable[indexS] == np.max(self.Qtable[indexS]))[0]) #cause sometimes two actions can have same q-value



class QLEARNING:
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes):
        self.env=env
        self.memory = deque(maxlen=2048)  # Deque is quick for append and pop operations from both the ends of the container.
        self.alpha=alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberEpisodes = numberEpisodes
        self.numberActions= self.env.action_space.n
        self.numberStates= self.env.observation_space.n
        #info to be saved
        self.reward_episodes = []
        self.observation_episodes = []
        self.step_episodes = []
        self.action_episodes=[]
        #state value
        self.learnedPolicy= np.zeros(self.env.observation_space.n)

        #Action value table

        self.Qtable= np.zeros((self.env.observation_space.n,self.env.action_space.n))

    # Add each step of an episode to agent's memory
    def add_memory(self, state, action, reward, new_state, terminated):
        # Insert the value in its argument to the right end of the deque
        self.memory.append((state, action, reward, new_state, terminated))


    def selectAction(self, state):
        #epsilon greedy policy

        if np.random.random()<self.epsilon:
            return np.random.choice(self.numberActions)

        else:
            return np.argmax(self.Qtable[state,:])

    def simulateEpisodes(self):

        # here we loop through the episodes
        for episode in range(self.numberEpisodes):

            # reset the environment at the beginning of every episode
            (currState, prob) = self.env.reset()
            # self.observation_cache.append(currState)
            state_arr = state_array(self.numberStates, currState)
            print("resetting environment for episode ", episode ," with initial state: " , currState," :" , prob)

            # select an action on the basis of the initial state
            currAction = self.selectAction(currState)

            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            while not terminalState:
                currAction = self.selectAction(currState)
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (nextState, reward, terminalState, _, _) = self.env.step(currAction)
                # self.observation_cache.append(nextState)
                new_state_arr = state_array(self.numberStates, nextState)
                # next action
                nextAction=np.argmax(self.Qtable[nextState,:])
                #print("nextAction",nextAction)

                if not terminalState:
                    error = reward + self.gamma * self.Qtable[nextState, nextAction] - self.Qtable[currState, currAction]
                    self.Qtable[currState, currAction] = self.Qtable[currState, currAction] + self.alpha * error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0
                    error = reward - self.Qtable[currState, currAction]
                    self.Qtable[currState, currAction] = self.Qtable[currState, currAction] + self.alpha * error
                self.reward_cache.append(reward)

                self.add_memory(state_arr, currAction, reward, new_state_arr, terminalState)
                currState = nextState
        save_data(self.memory,"QLEARNING")


    # now we compute the final learned policy
    def computeFinalPolicy(self):
            for indexS in range(self.numberStates):
                # we use np.random.choice() because in theory, we might have several identical maximums
                self.learnedPolicy[indexS] = np.random.choice(
                    np.where(self.Qtable[indexS] == np.max(self.Qtable[indexS]))[0]) #cause sometimes two actions can have same q-value


class DQN:
    def __init__(self,env,learning_rate,gamma,numberEpisodes,eps_decay):
        self.env=env
        self.memory = deque(maxlen=2048)  # Deque is quick for append and pop operations from both the ends of the container.
        self.learning_rate = learning_rate #0.001
        self.gamma = gamma
        self.numberEpisodes = 1001
        self.max_steps = 1000  # Maximum number of steps the agent can carry out in an episode
        self.numberActions= self.env.action_space.n
        self.numberStates= self.env.observation_space.n
        self.batch_size = 256
        #epsilon
        self.epsilon = 1  # Start (max) value of epsilon : ow random you take an action
        self.min_eps = 0.01  # End (min) value of epsilon
        self.eps_decay = eps_decay #0.0015 Decay factor for decreasing epsilon over episodes
        #model
        self.policy_model = self.build_model()  # Agent policy (and target model)
        self.target_model = self.build_model()
        self.update_target_model()
        #define edges
        self.left_edge_state = []
        self.top_edge_state = []
        self.right_edge_state = []
        self.bottom_edge_state = []
        rows = 12
        columns = 4
        for i in range(columns):
            state = i * 12
            state_right = i * 12 + rows - 1
            self.left_edge_state.append(state)
            self.right_edge_state.append(state_right)
        for i in range(rows):
            self.top_edge_state.append(i)
            state_bottom = 3 * 12 + i
            self.bottom_edge_state.append(state_bottom)


        #define allowed actions for edges
        self.action_left_edge = [0, 1, 2]
        self.action_top_edge = [1, 2, 3]
        self.action_right_edge = [0, 2, 3]
        self.action_bottom_edge = [0, 1, 3]

        #data to save
        self.reward_episodes = []
        self.observation_episodes = []
        self.step_episodes = []
        self.action_episodes=[]
        self.epsilon_episodes = []
        #state value
        self.learnedPolicy= np.zeros(self.env.observation_space.n)
        self.Qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    # Build model architecture and compile it
    def build_model(self):
        model = Sequential()  # Initiate a simple sequential model
        model.add(Dense(10, input_dim=self.numberStates, activation='relu'))
        # Add a fully connected layer
        # 10 is units : the size of the output from the dense layer
        # It's generally a good idea to start with a smaller number
        # of units and gradually increase them until you see improvement in the model's performance
        model.add(Dense(self.numberActions, activation='linear'))  # Add a fully connected layer connecting to the network output
        model.compile(loss='mse', optimizer=Adam( learning_rate=self.learning_rate))  # Compile model with Mean Squared Error loss function
        return model

    # Add each step of an episode to agent's memory
    def add_memory(self, state, action, reward, new_state, terminated):
        # Insert the value in its argument to the right end of the deque
        self.memory.append((state, action, reward, new_state, terminated))

    def replay(self):
        ### Process data batch
        # Select a data batch for training
        minibatch = random.sample(self.memory, self.batch_size)
        # Extract 'state' batch from the data batch
        minibatch_state = np.concatenate(np.array(minibatch, dtype=object)[:, 0],axis=0)
        # Extract 'action' batch from the data batch
        minibatch_action = np.array(minibatch, dtype=object)[:, 1]
        # Extract 'reward' batch from the data batch
        minibatch_reward = np.array(minibatch, dtype=object)[:, 2]
        # Extract 'new_state' batch from the data batch
        minibatch_new_state = np.concatenate(np.array(minibatch, dtype=object)[:, 3],axis=0)
        # Extract 'terminated' batch from the data batch
        minibatch_terminated = np.array(minibatch, dtype=object)[:,4]

        ### Feed state to the network for training

        # feed current state to the network to get current Q-values
        q_state = self.policy_model.predict(minibatch_state, verbose=0)
        # feed new state to the network to get next Q-values
        q_new_state = self.target_model.predict(minibatch_new_state, verbose=0)
        # Calculate the optimal Q-values of actions from the current state
        q_action_optimal = np.add(minibatch_reward, self.gamma * np.amax(q_new_state, axis=1))
        # Update optimal Q-values back to 'q_state' variable
        for i in range(0, self.batch_size):
            q_state[i][minibatch_action[i]] = q_action_optimal[i]
        # Train the network with the optimal Q-values, i.e., now in 'q_state' variable. x is minibatch_state and y is q_state.
        self.policy_model.fit(minibatch_state, q_state, epochs=1, verbose=0)


    # update target model with weights from policy model
    def update_target_model(self):
        self.target_model.set_weights(self.policy_model.get_weights())

    # Save model
    def save_model(self, name):
        self.policy_model.save_weights(name)

    # Load model
    def load_model(self, name):
        self.policy_model.load_weights(name)

    def define_edges(self, rows=12,columns=4):
        for i in range(columns):
            state = i * 12
            state_right = i * 12 + rows - 1
            self.left_edge_state.append(state)
            self.right_edge_state.append(state_right)
        for i in range(rows):
            self.top_edge_state.append(i)
            state_bottom = 3 * 12 + i
            self.bottom_edge_state.append(state_bottom)
        return self.left_edge_state, self.right_edge_state, self.top_edge_state, self.bottom_edge_state

    def check_action(self,state):

        if state[0] == 0:
            allowed_actions = [1, 2]
        elif state[0] == 11:
            allowed_actions = [2, 3]
        elif state[0] in self.left_edge_state:
            allowed_actions = self.action_left_edge
        elif state[0] in self.right_edge_state:
            allowed_actions = self.action_right_edge
        elif state[0] in self.top_edge_state:
            allowed_actions = self.action_top_edge
        elif state[0] in self.bottom_edge_state:
            allowed_actions = self.action_bottom_edge
        else:
            allowed_actions=np.arange(4)

        return allowed_actions

    def selectAction(self,state_arr,state):

        # 1st part exploit
        # Epsilon starts at 1. rand > epsilon: exploit, rand <= epsilon: explore.
        # explore more at first with epsilon 1 and exploit more at the end

        if np.random.rand() > self.epsilon:
            action_array = self.policy_model.predict(state_arr, verbose=0)[0]
            # print("predicted qvalues ", action_array)
            output_action = np.argmax(action_array)
            allowed_actions = self.check_action(state)
            #print("allowed actions: ", allowed_actions)
            # check if the output action in in allowed action
            while output_action not in allowed_actions:
                # Set the current output_action element to a very low value so it won't be chosen again
                # Then, find the argmax again to get the next highest element
                action_array[output_action] = -float('inf')
                output_action = np.argmax(action_array)
            # print("exploit action: ", output_action)

        else:
            output_action= random.choice(self.check_action(state))
            # print("explore action", output_action)
        return output_action


    def simulateEpisodes(self):

        for episode in range(self.numberEpisodes):
            # Set agent's current episode
            self.episode = episode
            state = self.env.reset()  # Reset environment, i.e., agent position is set to state 0
            state_arr = state_array(self.numberStates, state[0])
            reward = 0  # Initialize reward to zero
            terminated = False  # Initialize terminated to False
            reward_in_episode = []  # list to hold the sum of rewards in an episode

            for step in range(self.max_steps):
                action = self.selectAction(state_arr, state)
                # new state is an int of teh position where the agent went
                new_state, reward, terminated, info, _ = self.env.step(action)

                # this section changes the reward to make it easier for the model to converge
                # walking into cliff now gives -10 reward instead of 100, reaching the reward now gives
                # 200 instead of 0, making a step still gives -1
                if reward == -100:
                    reward = -10
                if terminated:
                    reward = 200

                new_state_arr = state_array(self.numberStates, state[0])
                reward_in_episode.append(reward)
                # Add current step to agent's memory
                self.add_memory(state_arr, action, reward, new_state_arr,terminated)
                # New state becomes state
                state_arr = new_state_arr


                if terminated:  # if agent reaches Goal
                    break  # End episode

            print("Episode: " + str(self.episode) + ", Steps: " + str(step) + ", epsilon: " + str(
                round(self.epsilon, 2)) + ", Reward: " + str(sum(reward_in_episode)))
            self.reward_episodes.append(reward)  # Add reward to list for tracking purpose
            self.step_episodes.append(step)  # Add step to list for tracking purpose
            self.epsilon_episodes.append(self.epsilon)  # Add epsilon to list for tracking purpose

            ### Update epsilon value which is the ratio of exploitation and exploration
            if self.epsilon < self.min_eps:
                self.epsilon = self.min_eps  # Set the minimum value of epsilon so that there is always a chance
                # for exploration.
            else:
                self.epsilon = np.exp(-self.eps_decay * self.episode)  # Decrease exploration / Increase exploitation
                # overtime as the agent knows more about the environment

            ### If enough experiences collected in the memory, use them for training
            if len(self.memory) > self.batch_size:
                self.replay()

            if self.episode % 10 == 0:
                self.update_target_model()

        save_data(self.memory, "DQN")
        print('Average success rate of training episodes: ', round(np.mean(self.reward_episodes), 2))
        self.save_model('model_' + str(self.episode))  # Save model


    # now we compute the final learned policy
    def computeFinalPolicy(self):
            self.load_model('model_' + str(self.numberEpisodes-1))
            for indexS in range(self.numberStates):
                # we use np.random.choice() because in theory, we might have several identical maximums
                state_arr = np.zeros(self.numberStates)
                # Set the agent position in the current state array, i.e., [[1. 0. 0. 0. 0. 0. 0. 0. 0.]]
                state_arr[indexS] = 1
                state_arr = np.reshape(state_arr, [1, self.numberStates])
                action_array = self.policy_model.predict(state_arr, verbose=0)[0]
                self.Qtable[indexS,:]=action_array
                output_action = np.argmax(action_array)
                self.learnedPolicy[indexS] = output_action
            # print(self.Qtable.shape)
            return self.learnedPolicy

