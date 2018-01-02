import numpy
import math
import random
import logging


def list_sum(li):
    res = 0
    for l in li:
        res += l
    return res


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
def get_train_data(filename: str):
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
def get_labels(filenanme: str):
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


# 获取训练集验证集划分的方式
def generate_split_list(amount: int, num: int):
    random.seed(1)
    order_list = [i for i in range(amount)]
    split_list = []
    split_size = amount // num
    for j in range(num-1):
        t_list = []
        for k in range(split_size):
            t_list.append(order_list.pop(random.randint(0, len(order_list)-1)))
        split_list.append(t_list)
    split_list.append(order_list)
    return split_list


# 将训练数据随机分成平均的几份
def split_dataset(data: list, traindata_list: list, valdata_list: list):
    """
    :param data: [array], 特征向量或标签[0,0,0]
    :param traindata_list: [int]，训练集的序号
    :param valdata_list: [int]， 验证集的序号
    :return: 训练集和验证集
    """
    random.seed(1)
    traindata_set = []
    valdata_set = []
    if len(traindata_list)+len(valdata_list) != len(data):
        print("splitting go wrong!")
    for k in traindata_list:
        traindata_set.append(data[k])
    for k in valdata_list:
        valdata_set.append(data[k])
    return traindata_set, valdata_set


# 获取训练集和验证集
def get_train_and_val(data_list: list, n: int):
    """
    :param data_list: list[list[int]], 分割后的数据集序号集合
    :param n: int, 第n份作为验证集序号集
    :return: list[int], list[int], 训练集序号和验证集序号
    """
    traindata_list = []
    for i in range(len(data_list)):
        if i != n:
            traindata_list += data_list[i]
    valdata_list = data_list[n]  # 剩下的一份作为验证集
    return traindata_list, valdata_list


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


def cal_confidence(x, w):
    confidence = numpy.dot(x, w)
    confidence = 1 / (1 + math.exp(-confidence))
    return confidence


def pre_one_label(x, w):
    predict_res = 0
    if cal_confidence(x, w) >= 0.5:
        predict_res = 1
    return predict_res


def pre_all_labels(x: numpy.ndarray, w: list):
    confidence = []
    for wi in w:
        confidence.append(cal_confidence(x, wi))
    predict_res = confidence.index(max(confidence))
    return predict_res


def pre_bagging(x_set: list, w_set: list, correction: list):
    predict_res = [0 for i in range(len(w_set[0]))]
    bagging_size = -1
    if len(x_set) == len(w_set):
        bagging_size = len(x_set)
    else:
        print("bagging x doesnt match w!")
    for i in range(bagging_size):
        pre = pre_all_labels(x_set[i], w_set[i])
        predict_res[pre] += correction[i]
    return predict_res.index(max(predict_res))


def val_one_label(valdata_x, valdata_y, w):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    correction = 0
    val_size = len(valdata_x)
    for vd in range(val_size):
        if pre_one_label(valdata_x[vd], w) == valdata_y[vd]:
            correction += 1
    return correction/val_size


def val_all_label(valdata_x, valdata_y: numpy.ndarray, w: list):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    correction = 0
    val_size = len(valdata_x)
    for td in range(val_size):
        real_label = valdata_y[td].tolist().index(1)
        if pre_all_labels(valdata_x[td], w) == real_label:
            correction += 1
    return correction / val_size


def val_bagging(data_x_set: list, data_y: numpy.ndarray, w_set: list, correction: list):
    bagging_correction = 0
    bagging_size = len(correction)
    data_size, temp = data_y.shape
    for i in range(data_size):
        x_set = []
        for k in range(bagging_size):
            x_set.append(data_x_set[k][i])
        if pre_bagging(x_set, w_set, correction) == data_y[i].tolist().index(1):
            bagging_correction += 1
    return bagging_correction/data_size


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
SFold = 5
Eta = 0.0001
IterTimes = 200
BaggingSize = 20
print("Eta=", Eta, "\nIterTimes=", IterTimes, "\nBaggingSize=", BaggingSize)
DataY = get_labels("label.txt")
# 规划训练集和验证集的划分
SplitList = generate_split_list(len(DataY), SFold)
TrainList, ValList = get_train_and_val(SplitList, 0)
Logger.info(TrainList)
Logger.info(ValList)
# 划分标签的训练和验证集
TrainDataY, ValDataY = split_dataset(DataY, TrainList, ValList)
TrainDataY = numpy.array(TrainDataY)
ValDataY = numpy.array(ValDataY)
Correction = []  # 之后用于加权的正确率
WSet = []  # 每次训练的三个w组成一个WSet项
TrainDataXSet = []  # 多次训练有多个训练集
ValDataXSet = []  # 和训练集对应的多个验证集
# 取出一份训练数据
for TrainSetIndex in range(BaggingSize):
    DataX = get_train_data("Train_onehot_"+str(TrainSetIndex)+".csv")
    TrainDataX, ValDataX = split_dataset(DataX, TrainList, ValList)
    TrainDataXSet.append(TrainDataX)
    ValDataXSet.append(ValDataX)
    W = []
    # 对三个标签都要训练
    for LabelIndex in range(3):
        W.append(train(TrainDataX, TrainDataY[:, LabelIndex], Eta, IterTimes))
    WSet.append(W)
    Correction.append(val_all_label(TrainDataX, TrainDataY, W))
    print("train correction:", Correction)
    print("val correction:", val_all_label(ValDataX, ValDataY, W))
# 使正确率和为1
print("使正确率和为1")
for CI in range(len(Correction)):
    Correction[CI] = Correction[CI]/list_sum(Correction)
print(Correction)
print("train after bagging: ", val_bagging(TrainDataXSet, TrainDataY, WSet, Correction))
print("val after bagging: ", val_bagging(ValDataXSet, ValDataY, WSet, Correction))
