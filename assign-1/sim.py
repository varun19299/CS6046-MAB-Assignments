import numpy as np 
from sacred import Experiment
from tupperware import tupperware
import matplotlib.pyplot as plt
import helper
from helper import Game
import sys

ex = Experiment("bandits")

@ex.config
def config():
    n = 10000        # event horizon
    repeat = 100      # repeat the experiment 100 times.
    games = [0,0,0,0]    # Bernouli distributed
    games[0] = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    games[1] = [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48]
    games[2] = [0.5, 0.2, 0.1]
    games[3] = [0.5, 0.4, 0.3, 0.42, 0.35, 0.22, 0.33]

def ETC(args, game_i, m):
    print(f"\n\nRunning game {game_i} with ETC of m = {m}")
    game_ll = args.games[game_i]
    print(f"Arms distribution used {game_ll}")
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)
    ##################################################### 
    # 0. Rewards collected and optimal arm pull (overall)
    #####################################################

    reward_ll = np.zeros(args.n)
    var_reward_ll = np.zeros(args.n)
    optimal_ll = np.zeros(args.n)
    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)

    for exp in range(args.repeat):
        ########################################### 
        # 0. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        reward_exp_ll = np.zeros(args.n)
        arm_exp_ll = np.zeros(args.n)
        ####################################### 
        # 1. Run through event horizon
        #######################################
        
        #####
        # ETC
        #####
        for k in range(m):
            for j in range(len(game)):
                # Explore
                arm_exp_ll[len(game)*k + j] = j
                reward = game.get_reward(j)
                reward_exp_ll[len(game)*k + j] = reward

        sample_mean_ll =  helper.get_sample_mean(reward_exp_ll, arm_exp_ll, n_arms = len(game), upto = m*len(game))
        best_sample_arm = np.argmax(sample_mean_ll)

        # print(f"\n\nExperiment {exp} : Sample mean table so far \n {sample_mean_ll}")
        # print(f"Experiment {exp} : Now exploit {best_sample_arm}")

        for j in range(m * len(game), args.n):
            # Exploit
            arm_exp_ll[j] = best_sample_arm
            reward = game.get_reward(best_sample_arm)
            reward_exp_ll[j] = reward

        regret_exp_ll = np.arange(1,args.n+1)*best_reward - np.cumsum(reward_exp_ll)
        reward_ll += reward_exp_ll
        var_reward_ll += reward_exp_ll**2
        optimal_ll += arm_exp_ll == best_arm
        regret_ll += regret_exp_ll
        var_regret_ll += regret_exp_ll**2

    reward_ll = reward_ll/ args.repeat
    regret_ll = regret_ll/ args.repeat
    var_reward_ll /= args.repeat
    var_reward_ll -= reward_ll**2
    var_reward_ll = np.sqrt(var_reward_ll)
    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll**2
    var_regret_ll /= np.sqrt(np.arange(1,args.n+1))
    var_regret_ll = np.sqrt(var_regret_ll)

    print("No of optimal pulls per trial",optimal_ll)
    print(f"Regret {regret_ll}")
    # print(np.cumsum(reward_ll))

    return reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll

def UCB(args, game_i, alpha = 0.2):
    print(f"\n\nRunning game {game_i} with UCB-alpha = {alpha}")
    game_ll = args.games[game_i]
    print(f"Arms distribution used {game_ll}")
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)
    ##################################################### 
    # 0. Rewards collected and optimal arm pull (overall)
    #####################################################

    reward_ll = np.zeros(args.n)
    var_reward_ll = np.zeros(args.n)
    optimal_ll = np.zeros(args.n)
    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)

    for exp in range(args.repeat):
        ########################################### 
        # 0. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        reward_exp_ll = np.zeros(args.n)
        arm_exp_ll = np.zeros(args.n)
        
        ##################################################
        # 1. UCB Initialisation
        ##################################################
        for j in range(len(game)):
            arm_exp_ll[j] = j                  
            reward = game.get_reward(j)
            reward_exp_ll[j] = reward

        sample_mean_ll =  helper.get_sample_mean(reward_exp_ll, arm_exp_ll, n_arms = len(game), upto = len(game))

        for j in range(len(game), args.n):
            # UCB = np.zeros(len(game))
            ##################################################
            # 2. Compute UCB metric
            ##################################################

            T_arm_ll = np.sum(arm_exp_ll[:j,None] == np.arange(len(game)), axis = 0)
            UCB = sample_mean_ll + np.sqrt(alpha*np.log(j+1)/T_arm_ll)
      
            # print(f"UCB {UCB}")
            # print(f"sample mean {sample_mean_ll}")
            ucb_arm = np.argmax(UCB)
            # print(f"Picking arm {ucb_arm}")
            arm_exp_ll[j] = ucb_arm
            reward = game.get_reward(ucb_arm)
            reward_exp_ll[j] = reward

            ##################################################
            # 3. Recompute Sample Mean
            ##################################################
            # sample_mean_ll =  helper.get_sample_mean(reward_exp_ll, arm_exp_ll, n_arms = len(game), upto = j + 2)
            sample_mean_ll[ucb_arm]+= (reward - sample_mean_ll[ucb_arm])/(T_arm_ll[ucb_arm]+1)

            # print(sample_mean_ll)
            # print(f"UCB {UCB}")

        # print(f"\n\nExperiment {exp} : UCB table so far \n {UCB}")
        reward_ll += reward_exp_ll
        var_reward_ll += reward_exp_ll**2
        optimal_ll += arm_exp_ll == best_arm
        regret_exp_ll = np.arange(1,args.n+1)*best_reward - np.cumsum(reward_exp_ll)
        regret_ll += regret_exp_ll
        var_regret_ll += regret_exp_ll**2

    # optimal_ll[990:] += 23*len(game)/10
    reward_ll = reward_ll/ args.repeat
    regret_ll = regret_ll/ (1*args.repeat)
    var_reward_ll /= args.repeat
    var_reward_ll -= reward_ll**2
    var_reward_ll = np.sqrt(var_reward_ll)
    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll**2
    var_regret_ll /= np.sqrt(np.arange(1,args.n+1))
    var_regret_ll = np.sqrt(var_regret_ll)

    print("No of optimal pulls per trial",optimal_ll)
    print(f"Regret {regret_ll}")
    # print(np.cumsum(reward_ll))

    return reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll

def plot_regret(D, game, args):
    '''
    D dictionary of algorithm names:
    (with keys)
    : regret 
    : var 
    '''
    l = np.arange(1,args.n+1)
    for alg in D:
        # plt.plot(l,D[alg]['regret'])
        plt.errorbar(l[::100],D[alg]['regret'][::100], D[alg]['var'][::100])
    
    plt.legend(list(D.keys()))
    plt.xlabel("Trial (n) Averaged over 100 experiments")
    plt.ylabel("Regret averaged over exps")
    plt.title(f"Regret Curves for game {game +1}")
    plt.savefig(f"Regret-game-{game +1}.png")
    plt.close()
    
def plot_opt(D, game, args):
    '''
    D dictionary of algorithm names:
    (with keys)
    : optimal 
    '''
    l = np.arange(1,args.n+1)
    for alg in D:
        plt.plot(l,D[alg]['optimal'],linewidth=1, alpha=0.6)
    
    plt.legend(list(D.keys()))
    plt.xlabel("Trial (n) Averaged over 100 experiments \n Note for ETC only shown for exploration.")
    plt.ylabel("Percentage of Optimal Arm pulls")
    plt.title(f"Percentage of Optimal Arm pulls vs rounds for game {game +1}")
    plt.savefig(f"Optimal-arm-{game+1}.png")
    plt.close()

@ex.automain 
def main(_run):
    args = _run.config
    print(f"Configs used {args}")
    args = tupperware(args)

    # ##################################################
    # # GAME 0
    # ##################################################

    D = {}
    for m in [783, 100,500,900]:
        reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll = ETC(args, 0, m=m)
        d={}
        d['regret'] = regret_ll
        d['var'] = var_regret_ll
        optimal_ll[:m*10] = 0
        d['optimal'] = optimal_ll
        D[f'ETC_m={m}'] = d
    for alpha in [0.5,2,5]:
        reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll = UCB(args, 0, alpha=alpha)
        d = {}
        d['regret'] = regret_ll
        d['var'] = var_regret_ll
        d['optimal'] = optimal_ll
        D[f'UCB-alpha={alpha}'] = d
        plot_regret(D, 0, args)
        plot_opt(D, 0, args)

    ##################################################
    # GAME 1
    ##################################################

    D = {}
    for m in [100,500,900]:
        reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll = ETC(args, 1, m=m)
        d={}
        d['regret'] = regret_ll
        d['var'] = var_regret_ll
        optimal_ll[:m*10] = 0
        d['optimal'] = optimal_ll
        D[f'ETC_m={m}'] = d
    for alpha in [0.5,2,5]:
        reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll = UCB(args, 1, alpha=alpha)
        d = {}
        d['regret'] = regret_ll
        d['var'] = var_regret_ll
        d['optimal'] = optimal_ll
        D[f'UCB-alpha={alpha}'] = d
        plot_regret(D, 1, args)
        plot_opt(D, 1, args)

    ##################################################
    # GAME 2
    ##################################################

    D = {}
    for m in [136, 50, 100]:
        reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll = ETC(args, 2, m=m)
        d={}
        d['regret'] = regret_ll
        d['var'] = var_regret_ll
        optimal_ll[:m*10] = 0
        d['optimal'] = optimal_ll
        D[f'ETC_m={m}'] = d
    for alpha in [0.5,2,5]:
        reward_ll, var_reward_ll, regret_ll, var_regret_ll, optimal_ll = UCB(args, 2, alpha=alpha)
        d = {}
        d['regret'] = regret_ll
        d['var'] = var_regret_ll
        d['optimal'] = optimal_ll
        D[f'UCB-alpha={alpha}'] = d
        plot_regret(D, 2, args)
        plot_opt(D, 2, args)