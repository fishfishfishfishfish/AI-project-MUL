import gensim
import random
import numpy
import math
import logging
import matplotlib.pyplot as plt


# 配置日志输出，方便debug
def get_logger():
    log_file = "./w2v_logger.log"
    log_level = logging.DEBUG

    logger = logging.getLogger("loggingmodule.NomalLogger")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("[%(levelname)s][%(funcName)s][%(asctime)s]%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


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


# 获取训练集和验证集
def get_train_and_val(data_list: list, n: int):
    traindata_list = []
    for i in range(len(data_list)):
        if i != n:
            traindata_list += data_list[i]
    valdata_list = data_list[n]  # 剩下的一份作为验证集
    return traindata_list, valdata_list


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


# 将训练数据切除验证集
def split_dataset(data: list, traindata_list: list, valdata_list: list):
    """
    :param data: [array], 特征向量或标签[0,0,0]
    :param traindata_list: [int]，训练集的序号
    :param valdata_list: [int]， 验证集的序号
    :return: 训练集和验证集
    """
    traindata_set = []
    valdata_set = []
    if len(traindata_list)+len(valdata_list) != len(data):
        print("splitting go wrong!")
    for k in traindata_list:
        traindata_set.append(data[k])
    for k in valdata_list:
        valdata_set.append(data[k])
    return traindata_set, valdata_set


def remove_null(sentences):
    for s in range(len(sentences)):
        while '' in sentences[s]:
            sentences[s].remove('')


def get_word_vec(model: gensim.models.word2vec.Word2Vec, sentence: list):
    res = numpy.zeros(model.vector_size)
    for word in sentence:
        if word in model:
            res += model[word]
    if len(sentence) > 0:
        res = res / len(sentence)
    return res


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


def cal_confidence(x, w):
    confidence = numpy.dot(x, w)
    confidence = 1 / (1 + math.exp(-confidence))
    return confidence


def pre_all_labels(x: numpy.ndarray, w: list):
    confidence = []
    for wi in w:
        confidence.append(cal_confidence(x, wi))
    predict_res = confidence.index(max(confidence))
    return predict_res


def val_all_label(valdata_x, valdata_y: numpy.ndarray, w: list):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    correction = 0
    val_size = len(valdata_x)
    for td in range(val_size):
        real_label = valdata_y[td].tolist().index(1)
        if pre_all_labels(valdata_x[td], w) == real_label:
            correction += 1
    return correction / val_size


def train(traindata_x, traindata_y, eta, iter_times):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    traindata_col = len(traindata_x[0])
    traindata_row = len(traindata_x)
    w = numpy.ones(traindata_col)
    last_likehood = -20
    cnt = 0
    record_likehood = []
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
            elif likehood - last_likehood > 10000:
                eta *= 1.1
                logger.debug("active2 eta=%f" % eta)
        last_likehood = likehood
        record_likehood.append(likehood)
        # logger.info(likehood)
        cnt += 1
    return w, record_likehood


def test(test_x, w, filename):
    res_string = ['LOW', 'MID', 'HIG']
    fout = open(filename, 'w')
    for x in test_x:
        res = pre_all_labels(x, w)
        fout.write(res_string[res] + '\n')


# 获取日志输出器
Logger = get_logger()
Eta = 0.0001
IterTimes = 200
SFold = 5
W2VIter = 100
W2VSize = 100
W2VWindow = 5
W2VName = "w2v_Iter" + str(W2VIter) + "&Size" + str(W2VSize) + "&Window" + str(W2VWindow)
print("LR : IterTimes = ", IterTimes)
print("w2v Model = ", W2VName)
TrainFile = open("text_train_out_withoutend.txt")
TestFile = open("text_test_out_withoutend.txt")
Labels = get_labels("label.csv")
TrainText = TrainFile.readlines()
TestText = TestFile.readlines()
TrainSentences = [line.strip('\n').split(' ') for line in TrainText]
TestSentences = [line.strip('\n').split(' ') for line in TestText]
remove_null(TrainSentences)
remove_null(TestSentences)
DataList = generate_split_list(len(TrainSentences), 5)
TrainList, ValList = get_train_and_val(DataList, 0)
TrainSentences, ValSentences = split_dataset(TrainSentences, TrainList, ValList)
TrainLabels, ValLabels = split_dataset(Labels, TrainList, ValList)
TrainLabels = numpy.array(TrainLabels)
ValLabels = numpy.array(ValLabels)
# 生成模型
Text = TrainText + TestText
Sentences = [line.split(' ') for line in Text]
Model = gensim.models.Word2Vec(Sentences, min_count=1, size=W2VSize, window=W2VWindow, iter=W2VIter)
Model.save(W2VName)
# 加载模型
# Model = gensim.models.Word2Vec.load(W2VName)
# Model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print("got model")
TrainX = []
ValX = []
for Sentence in TrainSentences:
    TrainX.append(get_word_vec(Model, Sentence))
for Sentence in ValSentences:
    ValX.append(get_word_vec(Model, Sentence))
W = []
LikeHoodRecords = []
for LabelsIndex in range(3):
    Logger.info(str(LabelsIndex)+"--------------------------")
    temp_w, temp_likehood = train(TrainX, TrainLabels[:, LabelsIndex], Eta, IterTimes)
    W.append(temp_w)
    LikeHoodRecords.append(temp_likehood)
LikeHoodRange = range(len(LikeHoodRecords[0]))
plt.plot(LikeHoodRange, LikeHoodRecords[0], 'b-', LikeHoodRange,
         LikeHoodRecords[1], 'g-', LikeHoodRange, LikeHoodRecords[2], 'r-')
plt.show()
print("train correction:", val_all_label(TrainX, TrainLabels, W))
print("val correction:", val_all_label(ValX, ValLabels, W))
TestX = []
# 取得测试数据的词向量
for Sentence in TestSentences:
    TestX.append(get_word_vec(Model, Sentence))
test(TestX, W, "w2v_LR_result.csv")