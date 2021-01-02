import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def rect_wave(x, c, c0):  # 起点为c0，宽度为c的矩形波
    if x >= (c + c0):
        r = 0.0
    elif x < c0:
        r = 0.0
    else:
        r = 1
    return r


def matched_filter(slogan1,slogan2,t0):
    datarate = 1 / 10
    fs = 100 * datarate
    fc = 1
    #t0 = 2
    t = np.linspace(0, 1 / datarate, int(fs / datarate))
    t1 = np.linspace(0, 2 / datarate, int(2 * fs / datarate - 1))
    f = np.linspace(-fs / 2, fs / 2, len(t1))
    sig = np.random.randn(len(t))  # 生成随机数
    if slogan1== 1:
        carrier = np.array([])#原始信号波形选择
        carrier = np.insert(carrier, len(carrier), np.cos(2 * (np.pi) * fc * t))
        h2 = np.array([])#正确匹配滤波波形选择
        h2 = np.insert(h2, len(h2), np.cos(2 * (np.pi) * fc * (2 - t)))
        a = np.sum(h2 ** 2)
        h2 = h2 / (a ** 0.5)#对传递函数能量归一化处理
    else:
        x = np.linspace(0, 2, 100)
        carrier = np.array([rect_wave(t, 1, 0) for t in x])
        h2=np.array([rect_wave(t,1,0) for t in 2-x])
        a = np.sum(h2 ** 2)
        h2 = h2 / (a ** 0.5)  # 对传递函数能量归一化处理

    if slogan2 == 1:
        h1 = np.array([])#自选匹配滤波波形
        h1 = np.insert(h1, len(h1), np.cos(2 * (np.pi) * fc * (t0 - t)))
        a = np.sum(h1 ** 2)
        h1 = h1 / (a ** 0.5)#对传递函数能量归一化处理
    else:
        x = np.linspace(0, 2, 100)
        h1 = np.array([rect_wave(t, 1, 0) for t in t0 - x])
        a = np.sum(h1 ** 2)
        h1 = h1 / (a ** 0.5)  # 对传递函数能量归一化处理
    autocorr1 = signal.convolve(carrier, h1, mode='full')  # 自选匹配滤波卷积结果
    autocorr2 = signal.convolve(carrier, h2, mode='full')  # 正确匹配滤波卷积结果
    result1 = autocorr1**2  # 自选匹配滤波信号瞬时能量
    result2 = autocorr2**2  # 正确匹配滤波信号瞬时能量
    return result1, result2, autocorr1, autocorr2, f
    '''
    fig, (ax_orig, ax_mag) = plt.subplots(2, 1)  # 建立两行一列图形
    ax_orig.plot(f, autocorr1**2)  # 自选匹配滤波信号瞬时能量
    ax_orig.set_title('自选匹配滤波信号瞬时能量')

    ax_mag.plot(f, autocorr2**2)  # 正确匹配滤波信号瞬时能量

    ax_mag.set_title('正确匹配滤波信号瞬时能量')  # 设置标题

    fig.tight_layout()  # 此句可以防止图像重叠
    fig.show()  # 显示图像
    
matched_filter(1,2,2)
'''