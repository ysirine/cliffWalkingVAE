import gym
import numpy as np
import time
import cv2
from agents import SARSA, QLEARNING, DQN
from utility import Env,generate_heatmap


# Create the Cliff Walking environment
env = gym.make('CliffWalking-v0')

# Reset the environment to its initial state
observation = env.reset()

# Set the number of steps to take
num_steps = 100
# step size
alpha = 0.1
# discount rate
gamma = 0.9
# epsilon-greedy parameter
epsilon = 0.2
eps_decay=0.0015
# number of simulation episodes
numberEpisodes = 10000
learning_rate = 0.001

width, height = 600, 200
margin_horizontal = 6
margin_vertical = 2
numVerticalLines=13
numHorizontalLines=5

                  ##########################      SARSA     ###########################
# initialize
SARSA1 = SARSA(env, alpha, gamma, epsilon, numberEpisodes)
# simulate
SARSA1.simulateEpisodes()
# compute the final policy
SARSA1.computeFinalPolicy()
# extract the final policy
finalLearnedPolicy = SARSA1.learnedPolicy
#show policy on gridworld
frameSARSA = Env(width, height,margin_horizontal,margin_vertical, numVerticalLines,numHorizontalLines)
frameSARSA.initialize_frame()
cv2.imshow("Cliff World SARSA", frameSARSA.plot_policy(finalLearnedPolicy))
cv2.waitKey(0)
#generate heatmap
generate_heatmap(SARSA1.Qtable)


#                   ##########################      QLEARNING     ###########################
#
# # initialize
# QLEARNING1= QLEARNING(env, alpha, gamma, epsilon, numberEpisodes)
# # simulate
# QLEARNING1.simulateEpisodes()
# # compute the final policy
# QLEARNING1.computeFinalPolicy()
# # extract the final policy
# finalLearnedPolicy1 = QLEARNING1.learnedPolicy
# #show policy on gridworld
# frameQLEARNING = Env(width, height,margin_horizontal,margin_vertical, numVerticalLines,numHorizontalLines)
# frameQLEARNING.initialize_frame()
# cv2.imshow("Cliff World QLEARNING", frameQLEARNING.plot_policy(finalLearnedPolicy1))
# cv2.waitKey(0)
# #generate heatmap
# generate_heatmap(QLEARNING1.Qtable)

                  ##########################      DQN     ###########################

# # initialize
# DQN1 = DQN(env,learning_rate, gamma, numberEpisodes,eps_decay)
# # simulate
# DQN1.simulateEpisodes()
# # compute the final policy
# DQN1.computeFinalPolicy()
# # extract the final policy
# finalLearnedPolicy2 = DQN1.learnedPolicy
# #show policy on gridworld
# frameDQN = Env(width, height,margin_horizontal,margin_vertical, numVerticalLines,numHorizontalLines)
# frameDQN.initialize_frame()
# cv2.imshow("Cliff World DQN", frameDQN.plot_policy(finalLearnedPolicy2))
# cv2.waitKey(0)
# #generate heatmap
# generate_heatmap(DQN1.Qtable)
