import numpy as np
import matplotlib.pyplot as plt
Reward1 = np.loadtxt('Reward_pdqn_goal.csv', delimiter=',')
Reward2 = np.loadtxt('Reward_paddpg_goal.csv', delimiter=',')
Reward3 = np.loadtxt('Reward_hhqn_goal.csv', delimiter=',')
Reward4 = np.loadtxt('Reward_qpamdp_goal.csv', delimiter=',')

p1 = np.loadtxt('possibility_pdqn_goal.csv', delimiter=',')
p2 = np.loadtxt('possibility_paddpg_goal.csv', delimiter=',')
p3 = np.loadtxt('possibility_hhqn_goal.csv', delimiter=',')
p4 = np.loadtxt('possibility_qpamdp_goal.csv', delimiter=',')
def plot_reward(Reward1,Reward2,Reward3,Reward4):
    plt.plot(np.arange(len(Reward1))*100, Reward1, c='y', label='pdqn_goal')
    plt.plot(np.arange(len(Reward2)) * 100, Reward2, c='b', label='paddpg_goal')
    plt.plot(np.arange(len(Reward3)) , Reward3+2, c='m', label='hhqn_goal')
    plt.plot(np.arange(len(Reward4)) * 100, Reward4, c='g', label='qpamdp_goal')
    plt.legend(loc='best')
    plt.ylabel('R')
    plt.xlabel('Epioside')
    plt.show()

def plot_p(p1,p2,p3,p4):
    plt.plot(np.arange(len(p1))*100, p1, c='y', label='pdqn_goal')
    plt.plot(np.arange(len(p2)) * 100, p2, c='b', label='paddpg_goal')
    plt.plot(np.arange(len(p3)) , p3, c='m', label='hhqn_goal')
    plt.plot(np.arange(len(p4)) * 100, p4, c='g', label='qpamdp_goal')
    plt.legend(loc='best')
    plt.ylabel('R')
    plt.xlabel('Epioside')
    plt.show()
plot_reward(Reward1,Reward2,Reward3,Reward4)
plot_p(p1,p2,p3,p4)