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
    :return: list[numpy.array[float]], 训练数据[x1,x2,x3,...]
    """
    data = []
    f = open(filename, 'r')
    lines = f.readlines()
    lines.pop(0)
    for line in lines:
        row = line.split(',')
        row.pop(len(row)-1)
        for i in range(len(row)):
            row[i] = float(row[i])
        row.insert(0, 1)
        row = numpy.array(row)
        data.append(row)
    return data


# 从文件读取训练集标签
def get_labels(filenanme):
    """
    :param filenanme: 训练文件名
    :return: list[array[0, 0, 0]], 标签对应的0/1表示
    """
    labels = []
    f = open(filenanme, 'r')
    for label in f.readlines():
        label = label.strip('\n')
        if label == "LOW":
            labels.append(numpy.array([1, 0, 0]))
        elif label == "MID":
            labels.append(numpy.array([0, 1, 0]))
        elif label == "HIG":
            labels.append(numpy.array([0, 0, 1]))
        else:
            print("labels go wrong!")
    return labels


# 将训练数据随机分成平均的几份
def split_dataset(data_x, data_y, num):
    """
    :param data_x: list[array[float]], 原始的训练集特征向量
    :param data_y: list[array[float]], 原始的训练集标签
    :param num: int, 要分成的份数
    :return: list[list[array[float]], 分解后的num个数据集集合
    """
    random.seed(1)
    data_x_list = []  # 由小份的数据集组成的
    data_y_list = []  # 由对应的小份标签组成的
    data_size = -1
    if len(data_x) != len(data_y):
        print("x and y don't match")
    else:
        data_size = len(data_x)
    val_size = data_size // num
    for k in range(num - 1):
        t_data_x = []
        t_data_y = []
        for i in range(val_size):
            cur_choice = random.randint(0, len(data_x) - 1)
            t_data_x.append(data_x.pop(cur_choice))
            t_data_y.append(data_y.pop(cur_choice))
        data_x_list.append(t_data_x)
        data_y_list.append(t_data_y)
    data_x_list.append(data_x)
    data_y_list.append(data_y)
    return data_x_list, data_y_list


# 获取训练集和验证集
def get_train_and_val(data_x_list, data_y_list, n):
    """
    :param data_x_list: list[list[array[float]], 分割后的数据集集合
    :param data_y_list: list[list[array[0,0,0]]], 分割后的标签集合
    :param n: int, 第n份作为验证集
    :return: list[array[float]], list[array[flost]], 训练集和验证集
    """
    traindata_x = []
    traindata_y = []
    for i in range(len(data_x_list)):
        if i != n:
            traindata_x += data_x_list[i]
            traindata_y += data_y_list[i]
    valdata_x = data_x_list[n]  # 剩下的一份作为验证集
    valdata_y = data_y_list[n]
    return traindata_x, traindata_y, valdata_x, valdata_y


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
    exp_xw = math.e ** xw
    return y*xw-math.log(1+exp_xw, math.e)


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


def train(traindata_x, traindata_y, eta, iter_times):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    traindata_col = len(traindata_x[0])
    traindata_row = len(traindata_x)
    w = numpy.ones(traindata_col)
    last_likehood = -20
    cnt = 0
    while cnt < iter_times:
        grad_sum = numpy.zeros(traindata_col)
        likehood = 0
        for d in range(traindata_row):
            # print(cnt, d)
            grad_sum += cal_gradient(traindata_x[d], traindata_y[d], w)
            likehood += cal_likehood(traindata_x[d], traindata_y[d], w)
        w = w - eta*grad_sum
        if cnt != 0:
            if likehood - last_likehood < 0.1:
                eta *= 0.9
                logger.debug("active1 eta=%f" % eta)
            elif likehood - last_likehood > 100:
                eta *= 1.1
                logger.debug("active2 eta=%f" % eta)
        last_likehood = likehood
        logger.info(likehood)
        cnt += 1
    return w


def val(valdata_x, valdata_y, w):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    correction = 0
    val_size = len(valdata_x)
    for vd in range(val_size):
        confidence = numpy.dot(valdata_x[vd], w)
        confidence = 1/(1 + math.exp(-confidence))
        logger.debug(confidence)
        predict_res = 0
        if confidence >= 0.5:
            predict_res = 1

        if predict_res == valdata_y[vd]:
            correction += 1
    return correction/val_size


def val_all(valdata_x, valdata_y, w):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    correction = 0
    val_size = len(valdata_x)
    for td in range(val_size):
        confidence = []
        for label_index in range(3):
            confid_t = numpy.dot(valdata_x[td], w[label_index])
            confidence.append(1 / (1 + math.exp(-confid_t)))
        res_index = confidence.index(max(confidence))
        curr_y = valdata_y[td].tolist()
        real_index = curr_y.index(max(curr_y))
        if res_index == real_index:
            correction += 1
    return correction / val_size


def test(w, test_data, filename):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    outfile = open(filename, 'w')
    test_size = len(test_data)
    for td in range(test_size):
        confidence = []
        for label_index in range(3):
            confid_t = numpy.dot(test_data[td], w[label_index])
            confidence.append(1 / (1 + math.exp(-confid_t)))
        res_index = confidence.index(max(confidence))
        if res_index == 0:
            outfile.write("LOW\n")
        elif res_index == 1:
            outfile.write("MID\n")
        elif res_index == 2:
            outfile.write("HIG\n")
        else:
            print("test encount wrong label!")


# 获取日志输出器
Logger = get_logger()
# ---------------------小数据集----------------------
# Data = get_train_data('small-train.csv')
# print(Data)
# train(Data, 1)
# -------读取训练数据，根据S折交叉验证切割-------------
DataX = get_train_data('Train_TFIDF_dense_Nor.csv')
DataY = get_labels("label.txt")
Test = get_train_data("Test_TFIDF_dense_Nor.csv")
Sfold = 5
DataListX, DataListY = split_dataset(DataX.copy(), DataY.copy(), Sfold)
# Logger.info("DataSet is : ")
# Logger.debug(numpy.array(DataSet))
# S份中取出一份作为验证集，其余构成训练集
for DataIndex in range(Sfold):
    TrainDataX, TrainDataY, ValDataX, ValDataY = get_train_and_val(DataListX, DataListY, DataIndex)
    TrainDataY = numpy.array(TrainDataY)
    ValDataY = numpy.array(ValDataY)
    DataYarray = numpy.array(DataY)
    Wt = []
    W = []
    for LabelIndex in range(3):
        Eta = 0.0001
        IterTimes = 200
        # if LabelIndex == 1:
        #     IterTimes = 200
        #     Eta = 0.00001
        Wt.append(train(TrainDataX, TrainDataY[:, LabelIndex], Eta, IterTimes))
        W.append(train(DataX, DataYarray[:, LabelIndex], Eta, IterTimes+50))
        print(Wt[LabelIndex])
        Logger.debug("-------------训练集反验证后的置信度-------------------")
        print(val(TrainDataX, TrainDataY[:, LabelIndex], Wt[LabelIndex]))
        print(val(TrainDataX, TrainDataY[:, LabelIndex], W[LabelIndex]))
        Logger.debug("-------------验证集验证后的置信度-------------------")
        print(val(ValDataX, ValDataY[:, LabelIndex], Wt[LabelIndex]))
        print(val(ValDataX, ValDataY[:, LabelIndex], W[LabelIndex]))
    print("train correction:", val_all(TrainDataX, TrainDataY, Wt))
    print("val correction:", val_all(ValDataX, ValDataY, Wt))
    print("train correction:", val_all(TrainDataX, TrainDataY, W))
    print("val correction:", val_all(ValDataX, ValDataY, W))
    test(W, Test, "Test_result.csv")
    break
