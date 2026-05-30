"""
This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 6-dimensional Dubins car
attack-defense problem.
"""
import sys
sys.path.append("/mnt/e_disk/me22b041/BTP_final")

import gym_reachability.gym_reachability.envs
import os
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch

from RARL.safety_shield import BasicSafetyShield, AdaptiveSafetyShield, RobustHJCBFSafetyShield 

from RARL.DDQNPE_novel import DDQNPursuitEvasion
from RARL.config import dqnConfig
from RARL.utils import save_obj

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

# == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd',
    type=str
)
parser.add_argument(
    "-ct", "--costType", help="cost type", default='sparse', type=str
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-cpf", "--cpf", help="consider pursuer failure set", action="store_true"
)

# car dynamics
parser.add_argument(
    "-cr", "--consRadius", help="constraint radius", default=1., type=float
)
parser.add_argument(
    "-tr", "--targetRadius", help="target radius", default=.5, type=float
)
parser.add_argument(
    "-turn", "--turnRadius", help="turning radius", default=.25, type=float
)
parser.add_argument("-s", "--speed", help="speed", default=.75, type=float)

# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=30000, type=int
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=100000,
    type=int
)
parser.add_argument(
    "-ut", "--updateTimes", help="#hyper-param. steps", default=20, type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=25000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check period", default=25000, type=int
)

# hyper-parameters
parser.add_argument(
    "-a", "--annealing", help="gamma annealing", action="store_true"
)
parser.add_argument(
    "-arc", "--architecture", help="NN architecture", default=[512, 512, 512],
    nargs="*", type=int
)
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-3, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.9, type=float
)
parser.add_argument(
    "-act", "--actType", help="activation type", default='Tanh', type=str
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)
parser.add_argument(
    "-tt", "--terminalType", help="terminal value", default='g', type=str
)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file", default='experiments', type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true"
)
parser.add_argument(
    "--shield",
    help="type of safety shield: none / basic / adaptive / robust",
    default="basic",
    type=str
)
parser.add_argument(
    "--start_config", type=int, default=0, help="Config index start from"
)

class ShieldedEnv:
    def __init__(self, env, shield=None):
        self.env = env
        self.shield = shield
        self.numActionList = env.numActionList

    def step(self, action):

        state = self.env.state.copy()

        # split joint action
        evader_action = action // self.numActionList[1]
        pursuer_action = action % self.numActionList[1]

        # apply shield ONLY to evader
        if self.shield is not None:
            evader_action = self.shield.get_safe_evader_action(
                state, evader_action
            )

        # recombine action
        new_action = evader_action * self.numActionList[1] + pursuer_action

        evader_action = new_action // self.numActionList[1]
        pursuer_action = new_action % self.numActionList[1]

        return self.env.step((evader_action, pursuer_action))

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    
configs = [

# ===== BASELINE =====
# {"cr":1.0, "tr":0.5, "turn":0.25, "g":0.9999, "lr":1e-3},

# # ===== EASY CASES =====
# {"cr":0.5, "tr":0.7, "turn":0.2, "g":0.9999, "lr":1e-3},
# {"cr":0.6, "tr":0.6, "turn":0.2, "g":0.9999, "lr":1e-3},

# # ===== HARD CASES =====
# {"cr":1.5, "tr":0.3, "turn":0.4, "g":0.9999, "lr":1e-3},
# {"cr":1.2, "tr":0.4, "turn":0.35, "g":0.9999, "lr":1e-3},

# # ===== TURNING EFFECT =====
# {"cr":1.0, "tr":0.5, "turn":0.15, "g":0.9999, "lr":1e-3},  # very agile
# {"cr":1.0, "tr":0.5, "turn":0.5,  "g":0.9999, "lr":1e-3},  # poor turning

# ===== GAMMA EFFECT =====
{"cr":1.0, "tr":0.5, "turn":0.25, "g":0.99,   "lr":1e-3},
# {"cr":1.0, "tr":0.5, "turn":0.25, "g":0.95,   "lr":1e-3},

# ===== LR STABILITY =====
# {"cr":1.0, "tr":0.5, "turn":0.25, "g":0.9999, "lr":5e-4},

]
args = parser.parse_args()
print(args)

print(f"\nStarting from config index {args.start_config}\n")
for exp_id, cfg in enumerate(configs):
    if exp_id + 1 < args.start_config:
        continue
    print("\n==============================")
    print(f"Running config {exp_id+1} of {len(configs)}")
    print(cfg)
    print("==============================")

    # ===== OVERRIDE PARAMETERS =====
    args.consRadius = cfg["cr"]
    args.targetRadius = cfg["tr"]
    args.turnRadius = cfg["turn"]
    args.gamma = cfg["g"]
    args.learningRate = cfg["lr"]
    
    # == CONFIGURATION ==
    env_name = "dubins_car_novel-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maxUpdates = args.maxUpdates
    updateTimes = args.updateTimes
    updatePeriod = int(maxUpdates / updateTimes)
    updatePeriodHalf = int(updatePeriod / 2)
    maxSteps = 200

    name = f"CR{args.consRadius}_TR{args.targetRadius}_TURN{args.turnRadius}_G{args.gamma}_LR{args.learningRate}"
    
    # SELECT SHIELD TYPE
    shield_type = args.shield   # change: none/ basic/ adaptive/ robust
    
    if shield_type not in ["none", "basic", "adaptive", "robust"]:
        raise ValueError(f"Invalid shield type: {shield_type}")
    
    outFolder = os.path.join(args.outFolder, f'car-pe-DDQN_{shield_type}_10Literation_config_8', name)
    print(outFolder)
    figureFolder = os.path.join(outFolder, 'figure')
    os.makedirs(figureFolder, exist_ok=True)
    print("ENV NAME USED:", env_name)
    # == Environment ==
    print("\n== Environment Information ==")

    # print("REGISTERED ENVS:")
    # print([k for k in gym.envs.registry.keys() if "dubins" in k])
    # env = gym.make(
    #     env_name, device=device, mode='RA', doneType=args.doneType,
    #     sample_inside_obs=False, considerPursuerFailure=args.cpf
    # )
    
    base_env = gym.make(
        env_name,
        device=device,
        mode='RA',
        doneType=args.doneType,
        sample_inside_obs=False,
        considerPursuerFailure=args.cpf
    )

    # Correct parameter mapping
    base_env.set_constraint(radius=args.consRadius, car='both')
    base_env.set_target(radius=args.targetRadius, car='evader')  
    base_env.set_radius_rotation(R_turn=args.turnRadius)
    base_env.report()
    
    if shield_type == "basic":
        shield = BasicSafetyShield(base_env)
    elif shield_type == "adaptive":
        shield = AdaptiveSafetyShield(base_env)
    elif shield_type == "robust":
        shield = RobustHJCBFSafetyShield(base_env)
    else:
        shield = None

    # wrap environment
    env = ShieldedEnv(base_env, shield=None)

    stateDim = env.state.shape[0]
    actionNum = env.action_space.n
    action_list = np.arange(actionNum)
    env.set_seed(args.randomSeed)
    env.report()

# == Get and Plot max{l_x, g_x} ==
    if args.plotFigure or args.storeFigure:
        nx, ny = 101, 101
        theta, thetaPursuer = 0., 0.
        v = np.zeros((4, nx, ny))
        l_x = np.zeros((4, nx, ny))
        g_x = np.zeros((4, nx, ny))
        xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
        ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

        xPursuerList = [.1, .3, .5, .8]
        yPursuerList = [.1, .3, .5, .8]
    for i, (xPursuer, yPursuer) in enumerate(zip(xPursuerList, yPursuerList)):
        it = np.nditer(l_x[0], flags=['multi_index'])

        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]

            state = np.array([x, y, theta, xPursuer, yPursuer, thetaPursuer])
            l_x[i][idx] = env.target_margin(state)
            g_x[i][idx] = env.safety_margin(state)

            v[i][idx] = np.maximum(l_x[i][idx], g_x[i][idx])
            it.iternext()

    axStyle = env.get_axes()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        xPursuer = xPursuerList[i]
        yPursuer = yPursuerList[i]
        f = ax.imshow(
            v[i].T, interpolation='none', extent=axStyle[0], origin="lower",
            cmap="seismic", vmin=-1, vmax=1
        )
        env.plot_target_failure_set(ax, xPursuer=xPursuer, yPursuer=yPursuer)
        if i == 3:
            fig.colorbar(
                f, ax=ax, pad=0.01, fraction=0.05, shrink=0.95, ticks=[-1, 0, 1]
            )
        env.plot_formatting(ax)
    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

# == Agent CONFIG ==
    print("\n== Agent Information ==")
    if args.annealing:
        GAMMA_END = 0.9999
        EPS_PERIOD = int(updatePeriod / 10)
        EPS_RESET_PERIOD = updatePeriod
    else:
        GAMMA_END = args.gamma
        EPS_PERIOD = updatePeriod
        EPS_RESET_PERIOD = maxUpdates

    CONFIG = dqnConfig(
        DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
        MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
        MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
        ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
        GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
        EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
        LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=50
    )
    # print(vars(CONFIG))

    # == AGENT ==
    numActionList = env.numActionList
    numJoinAction = int(numActionList[0] * numActionList[1])
    dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
    agent = DDQNPursuitEvasion(
        CONFIG, numActionList, dimList, mode='RA', terminalType='g'
    )
    print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
    print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

    vmin = -1
    vmax = 1
    if args.warmup:
        print("\n== Warmup Q ==")
        lossList = agent.initQ(
            env, args.warmupIter, outFolder, num_warmup_samples=250, vmin=vmin,
            vmax=vmax, plotFigure=args.plotFigure, storeFigure=args.storeFigure
        )

        if args.plotFigure or args.storeFigure:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            tmp = np.arange(500, args.warmupIter)
            ax.plot(tmp, lossList[tmp], 'b-')

            ax.set_xlim(500, args.warmupIter)
            ax.set_xlabel('Iteration', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            ax.xaxis.set_major_locator(LinearLocator(5))
            ax.xaxis.set_major_formatter('{x:.1f}')
            ax.yaxis.set_major_locator(LinearLocator(5))
            ax.yaxis.set_major_formatter('{x:.1f}')
            fig.tight_layout()

        if args.storeFigure:
            figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
            fig.savefig(figurePath)
        if args.plotFigure:
            plt.show()
            plt.pause(0.001)
        plt.close()

    print("\n== Training Information ==")
    
    vmin = -1
    vmax = 1
    
    env = ShieldedEnv(base_env, shield=shield)   # switch ON shield
    
    trainRecords, trainProgress = agent.learn(
        env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
        doneTerminate=True, vmin=vmin, vmax=vmax, showBool=False,
        checkPeriod=args.checkPeriod, outFolder=outFolder,
        plotFigure=args.plotFigure, storeFigure=args.storeFigure
    )

    trainDict = {}
    trainDict['trainRecords'] = trainRecords
    trainDict['trainProgress'] = trainProgress
    filePath = os.path.join(outFolder, 'train')

    if args.plotFigure or args.storeFigure:
        # region: loss
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        data = trainRecords
        ax = axes[0]
        ax.plot(data, 'b:')
        ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
        ax.set_xticks(np.linspace(0, maxUpdates, 5))
        ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
        ax.set_title('loss_critic', fontsize=18)
        ax.set_xlim(left=0, right=maxUpdates)

        data = trainProgress[:, 0]
        ax = axes[1]
        x = np.arange(data.shape[0]) + 1
        ax.plot(x, data, 'b-o')
        ax.set_xlabel('Index', fontsize=18)
        ax.set_xticks(x)
        ax.set_title('Success Rate', fontsize=18)
        ax.set_xlim(left=1, right=data.shape[0])
        ax.set_ylim(0, 0.8)

        fig.tight_layout()
        if args.storeFigure:
            figurePath = os.path.join(figureFolder, 'train_loss_success.png')
            fig.savefig(figurePath)
        if args.plotFigure:
            plt.show()
            plt.pause(0.001)
        plt.close()
    # endregion

    # region: value_rollout_action
    idx = np.argmax(trainProgress[:, 0]) + 1
    successRate = np.amax(trainProgress[:, 0])
    print('We pick model with success rate-{:.3f}'.format(successRate))
    agent.restore(idx * args.checkPeriod, outFolder)

    nx = 101
    ny = nx
    xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
    ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

    resultMtx = np.empty((nx, ny), dtype=int)
    actDistMtx = np.empty((nx, ny), dtype=int)
    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y, 0., -0.2, -0.3, 0.75 * np.pi])
        stateTensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
        state_action_values = agent.Q_network(stateTensor)
        Q_mtx = state_action_values.reshape(
            env.numActionList[0], env.numActionList[1]
        )
        # Q_mtx = Q_mtx.detach().cpu()
        pursuerValues, colIndices = Q_mtx.max(dim=1)
        _, rowIdx = pursuerValues.min(dim=0)
        colIdx = colIndices[rowIdx]

        uEvader = env.evader.discrete_controls[rowIdx]
        uPursuer = env.pursuer.discrete_controls[colIdx]
        actDistMtx[idx] = rowIdx

        _, _, result, _, _ = env.simulate_one_trajectory(
            agent.Q_network, T=250, state=state, toEnd=False
        )
        resultMtx[idx] = result
        it.iternext()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axStyle = env.get_axes()

    # = Action
    ax = axes[2]
    im = ax.imshow(
        actDistMtx.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap='seismic', vmin=0, vmax=actionNum - 1, zorder=-1
    )
    ax.set_xlabel('Action', fontsize=24)

    # = Rollout
    ax = axes[1]
    im = ax.imshow(
        resultMtx.T != 1, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1
    )
    env.plot_trajectories(
        agent.Q_network, states=[env.visual_initial_states[1]], toEnd=False,
        ax=ax, lw=1.5, T=200
    )
    ax.set_xlabel('Rollout RA', fontsize=24)

    # = Value
    ax = axes[0]
    im = ax.imshow(
        v.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1
    )
    ax.set_xlabel('Value', fontsize=24)

    for ax in axes:
        env.plot_target_failure_set(ax=ax, xPursuer=-0.2, yPursuer=-0.3)
        env.plot_formatting(ax=ax)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    # endregion

    trainDict['resultMtx'] = resultMtx
    trainDict['actDistMtx'] = actDistMtx

    save_obj(trainDict, filePath)