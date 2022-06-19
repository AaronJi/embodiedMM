import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_index_age():
    d = np.array(
        [
            [4.73, 1.35, -1.42, -2.82],
            [4.36, 0.92, -0.69, -3.27],
            [2.38, -0.64, -0.80, -4.77],
            [2.04, -0.16, -0.99, -5.31],
            [-4.66, -2.72, -5.09, -8.90],
            [-4.28, -4.78, -0.97, -8.33],
            [2.18, -1.63, -3.44, -5.11]
        ]
    ) / 100.0

    print(d)

    plt.figure(0)
    # plt.plot(d)
    k = 6
    plt.plot(d[:k, 0], '-or')
    plt.plot(d[:k, 1], '-ok')
    plt.plot(d[:k, 2], '-oc')
    plt.plot(d[:k, 3], '-ob')
    plt.legend(['Empirical', 'LSTM', 'RL-LTV-woR', 'RL-LTV'])
    plt.xlabel('Recommendation Position Group', fontsize=14)
    plt.ylabel('Percentage Difference of Mean Item Age', fontsize=14)

    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})

    # plt.axis([0, 1, 2, 3, 4], [-0.02, 0, 0.02])
    plt.tick_params(axis='both', which='major', labelsize=10)
    x_major_locator = plt.MultipleLocator(1)
    y_major_locator = plt.MultipleLocator(0.04)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    def to_percent(temp, position):
        return '%0.00f' % (100 * temp) + '%'

    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_xticks(range(k))
    ax.set_xticklabels(['1-10', '11-20', '21-30', '31-40', '41-50', '51-100'], rotation=0)

    # plt.xlim(0, 4)
    plt.show()


def plot_alpha_sensitivty():
    alpha = 0.1 * np.array(range(11))

    d = np.array(
        [
            [0.7342625067609376, 0.722,0.655,0.608],
            [0.7286495415666523, 0.723,0.658,0.610],
            [0.7132369478053959, 0.727,0.661,0.613],
            [0.6902906715292065, 0.730,0.664,0.614],
            [0.6623187359185296, 0.734,0.667,0.616],
            [0.6317164962475843, 0.741,0.674,0.622],
            [0.6009110837713063, 0.746,0.679,0.626],
            [0.5719084800913722, 0.749,0.684,0.631],
            [0.5455840474362383, 0.750,0.686,0.633],
            [0.5220599026608743, 0.751,0.688,0.636],
            [0.5009771639609544, 0.752,0.690,0.638]
        ]
    )

    print(d)

    plt.figure(0)
    # plt.plot(d)
    k = 6
    plt.plot(alpha[:k], d[:k, 0], linestyle = '-',marker = "s",color = "cornflowerblue",linewidth = 3)
    plt.plot(alpha[:k], d[:k, 1], linestyle = '-',marker = "^",color = "mediumpurple",linewidth = 3)
    plt.plot(alpha[:k], d[:k, 2], linestyle = '-',marker = "o",color = "sandybrown",linewidth = 3)
    plt.plot(alpha[:k], d[:k, 3], linestyle = '-',marker = "D",color = "lightcoral",linewidth = 3)
    plt.legend(['AUC', 'NDCG@10','NDCG@20','NDCG@50'],fontsize=8)
    plt.xlabel(r'$\alpha$', fontsize=18)
    # plt.ylabel('Metric', fontsize=18)

    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})

    # plt.axis([0, 1, 2, 3, 4], [-0.02, 0, 0.02])
    plt.tick_params(axis='both', which='major', labelsize=10)
    x_major_locator = plt.MultipleLocator(0.1)
    # y_major_locator = plt.MultipleLocator(0.04)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

    plt.show()


def plot_item_growth():
    pv_acc = np.array([
        0,
        1176,
        2855,
        3064,
        3196,
        3248,
        3341,
        3903,
        5598,
        7742,
        9031,
        9734,
        10336,
        16189
    ])
    ipv_acc = np.array([
        4,
        74,
        173,
        338,
        493,
        701,
        962,
        1222,
        1529,
        1735,
        2021,
        2317,
        2663,
        2950
    ])
    gmv_acc = np.array([
        0,
        0,
        28.56,
        132.68,
        170.36,
        231,
        369.94,
        520.04,
        715.77,
        835.23,
        910.58,
        1141.49,
        1301.2,
        1379.2
    ])
    pv_rec = np.array([
        0,
        1175,
        1678,
        208,
        132,
        34,
        55,
        524,
        1667,
        2113,
        1234,
        612,
        511,
        5795,
    ])

    pv = np.array([
        0,
        1176,
        1679,
        209,
        132,
        52,
        93,
        562,
        1695,
        2144,
        1289,
        703,
        602,
        5853
    ])

    ipv = np.array([
        4,
        70,
        99,
        165,
        155,
        208,
        261,
        260,
        307,
        206,
        286,
        296,
        346,
        287
    ])

    gmv = np.array([
        0,
        0,
        28.56,
        104.12,
        37.68,
        60.64,
        138.94,
        150.1,
        195.73,
        119.46,
        75.35,
        230.91,
        159.71,
        78
    ])

    n = len(pv_acc)
    '''
    pv = np.zeros(n)
    for i in range(n-1):
        pv[i] = pv_acc[i+1] - pv_acc[i]
    pv[-1] = 50
    print(pv)    
    '''

    # d = np.vstack((pv_acc, ipv_acc, gmv_acc, pv_rec, pv)).transpose()
    # print(d)
    k = 8

    plt.figure(0)

    # plt.subplot(2, 2, 1)

    plt.subplot(2, 2, 2)
    plt.plot(pv_rec[:k], '-ob')
    # plt.legend(['rec'])
    # plt.xlabel('age', fontsize=10)
    # plt.ylabel(r'$PV_{rec}$', fontsize=10)
    plt.legend([r'$PV_{rec}$'], fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    x_major_locator = plt.MultipleLocator(1)
    # y_major_locator = plt.MultipleLocator(0.04)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

    plt.subplot(2, 2, 3)

    plt.plot(pv_acc[:k], '-ob')
    plt.plot(ipv_acc[:k], '-oy')
    plt.plot(gmv_acc[:k], '-og')
    plt.legend([r'$PV_{acc}$', r'$IPV_{acc}$', r'$GMV_{acc}$'], fontsize=8)
    plt.xlabel('Time on the market', fontsize=10)
    # plt.ylabel('accumulated metric', fontsize=14)

    plt.tick_params(axis='both', which='major', labelsize=8)
    x_major_locator = plt.MultipleLocator(1)
    # y_major_locator = plt.MultipleLocator(0.04)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

    plt.subplot(2, 2, 4)
    plt.plot(pv[:k] - pv_rec[:k], '-oc')
    plt.legend([r'$PV_{other}$'])
    plt.xlabel('Time on the market', fontsize=10)
    # plt.ylabel(r'$PV_{other}$', fontsize=10)
    plt.legend([r'$PV_{other}$'], fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    x_major_locator = plt.MultipleLocator(1)
    # y_major_locator = plt.MultipleLocator(0.04)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

    '''
    plt.plot(pv[:k], '-ob')
    plt.plot(ipv[:k], '-oy')
    plt.plot(gmv[:k], '-og')
    plt.legend([r'$PV$', r'$IPV$', r'$GMV$'], fontsize=8)
    plt.xlabel('age', fontsize=10)

    plt.tick_params(axis='both',which='major',labelsize=8)
    x_major_locator = plt.MultipleLocator(1)
    #y_major_locator = plt.MultipleLocator(0.04)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)    
    '''

    plt.show()
    return

def plot_traj_result():
    num_trajectory = np.array([
        10,
        100,
        1000
    ])
    ave_reward = np.array([
        1.797,
        39.233,
        99.678
    ])
    ave_length = np.array([
        6,
        22,
        56
    ])
    dt_baseline = [1000, 102, 42]
    plt.subplot(1, 2, 1)
    plt.plot(num_trajectory, ave_reward, '-og')
    plt.scatter(dt_baseline[0], dt_baseline[1])
    plt.legend([r'decision OFA', 'Decision Transformer'], fontsize=8)
    plt.xlabel('num of training trajectories', fontsize=10)
    plt.ylabel('ave reward', fontsize=10)
    plt.subplot(1, 2, 2)
    plt.plot(num_trajectory, ave_length, '-og')
    plt.scatter(dt_baseline[0], dt_baseline[2])
    plt.legend([r'decision OFA', 'Decision Transformer'], fontsize=8)
    plt.xlabel('num of training trajectories', fontsize=10)
    plt.ylabel('ave episode len', fontsize=10)

    plt.show()
    return

if __name__ == "__main__":
    # plot_index_age()
    #plot_alpha_sensitivty()
    # plot_item_growth()
    plot_traj_result()
