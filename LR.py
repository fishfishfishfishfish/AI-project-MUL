import numpy
import math
import random
import logging


# 配置日志输出，方便debug
def get_logger():
    log_file = "./nomal_logger.log"
    log_level = logging.DEBUG

    logger = logging.getLogger("loggingmodule.NomalLogger")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("[%(levelname)s][%(funcName)s][%(asctime)s]%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# 从文件读取训练数据
def get_train_data(filename):
    """
    :param filename: string, 训练文件名
    :return: list[numpy.array[float]], 训练数据[x,x,x,...,y]
    """
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        row = line.split(',')
        for i in range(len(row)):
            row[i] = float(row[i])
        row.insert(0, 1)
        row = numpy.array(row)
        data.append(row)
    return data

# 从文件读取训练集标签
def get_labels(filenanme):
    f = open(filenanme, 'r')
    for label in f.readlines():
        pass


# 将训练数据随机分成平均的几份
def split_dataset(data, num):
    """
    :param data: list[numpy.array[float]], 原始的训练集
    :param num: int, 要分成的份数
    :return: list[list[array[float]], 分解后的num个数据集数组
    """
    random.seed(1)
    data_list = []
    val_size = len(data) // num
    for k in range(num - 1):
        t_data = []
        for i in range(val_size):
            t_data.append(data.pop(random.randint(0, len(data) - 1)))
        data_list.append(t_data)
    data_list.append(data)
    return data_list


# 获取训练集和验证集
def get_train_and_val(dataset, n):
    """
    :param dataset: list[list[array[float]], 分割后的数据集
    :param n: int, 第n份作为验证集
    :return: list[array[float]], list[array[flost]], 训练集和验证集
    """
    traindata = []
    for i in range(len(dataset)):
        if i != n:
            traindata += dataset[i]
    valdata = dataset[n]  # 剩下的一份作为验证集
    return traindata, valdata


# 计算对数似然值
def cal_likehood(x, y, w):
    """
    :param x: numpy vector, 增广特征向量
    :param y: int, x对应的结果
    :param w: numpy vector, 增广权向量
    :return: int, 对数似然值
    """
    x = numpy.array(x)
    w = numpy.array(w)
    xw = numpy.dot(x, w)
    exp_xw = math.e ** (xw)
    return y*xw-math.log(1+exp_xw,math.e)


# 计算一行x和当前w计算得到的梯度值
def cal_gradient(x, y, w):
    """
    :param x: numpy vector, 增广特征向量
    :param y: int, x对应的结果
    :param w: numpy vector, 增广权向量
    :return: 一行数据的梯度
    """
    logger = logging.getLogger("loggingmodule.NomalLogger")
    x = numpy.array(x)
    w = numpy.array(w)
    nxw = -numpy.dot(x, w)
    exp_xw = math.exp(nxw)
    return (1/(1+exp_xw)-y)*x


def train(traindata, eta):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    w = numpy.ones(len(traindata[0])-1)
    last_likehood = -20
    cnt = 0
    while cnt < 5000:
        grad_sum = numpy.zeros(len(traindata[0])-1)
        likehood = 0
        for d in traindata:
            grad_sum += cal_gradient(d[0:len(d)-1], d[len(d)-1], w)
            likehood += cal_likehood(d[0:len(d)-1], d[len(d)-1], w)
        w = w - eta*grad_sum
        if cnt != 0:
            if likehood < last_likehood:
                eta -= 0.0000005
                logger.debug("active1 eta=%f" % eta)
            elif likehood - last_likehood < 0.1 and likehood - last_likehood > 0:
                eta += 0.0000001
                logger.debug("active2 eta=%f" % eta)
        last_likehood = likehood
        logger.info(likehood)
        cnt += 1
    return w


def val(valdata, w):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    correction = 0
    for vd in valdata:
        confidence = numpy.dot(vd[0:len(vd)-1], w)
        confidence = 1/(1 + math.exp(-confidence))
        logger.debug(confidence)
        predict_res = 0
        if confidence >= 0.5:
            predict_res = 1

        if predict_res == vd[len(vd)-1]:
            correction += 1
    return correction/len(valdata)


# 获取日志输出器
Logger = get_logger()
# ---------------------小数据集----------------------
# Data = get_train_data('small-train.csv')
# print(Data)
# train(Data, 1)
# -------读取训练数据，根据S折交叉验证切割-------------
Data = get_train_data('train.csv')
Sfold = 10
DataSet = split_dataset(Data, Sfold)
# Logger.info("DataSet is : ")
# Logger.debug(numpy.array(DataSet))
# S份中取出一份作为验证集，其余构成训练集
for k in range(Sfold):
    TrainData, ValData = get_train_and_val(DataSet, k)
    W = train(TrainData, 0.000011)
    print(W)
    Logger.debug("-------------训练集反验证后的置信度-------------------")
    print(val(TrainData, W))
    Logger.debug("-------------验证集验证后的置信度-------------------")
    print(val(ValData, W))
    break