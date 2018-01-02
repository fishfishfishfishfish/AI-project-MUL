import numpy
import logging
import random
import gensim


# 配置日志输出，方便debug
def get_logger():
    log_file = "./w2v_softmax_logger.log"
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
            labels.append(0)
        elif label == "MID":
            labels.append(1)
        elif label == "HIG":
            labels.append(2)
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
    if len(traindata_list) + len(valdata_list) != len(data):
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


# 获取词向量
def get_word_vec(model: gensim.models.word2vec.Word2Vec, sentence: list):
    res = numpy.zeros(model.vector_size)
    for word in sentence:
        if word in model:
            res += model[word]
    if len(sentence) > 0:
        res = res / len(sentence)
    return res


def classify(traindata, trainlabel, testdata: numpy.ndarray, k):
    train_size = len(traindata)
    dist_vec = numpy.linalg.norm(numpy.mat(traindata) - numpy.mat(testdata), axis=-1)
    dist_list = []
    for i in range(train_size):
        dist = (dist_vec[i], trainlabel[i])
        dist_list.append(dist)
    dist_list = sorted(dist_list)
    label_list = [0, 0, 0]
    for i in range(k):
        label_list[dist_list[i][1]] += 1
    return label_list.index(max(label_list))


def val(traindata, trainlabel, valdata, vallabel, k):
    correction = 0
    for i in range(len(valdata)):
        # print(classify(traindata, trainlabel, valdata[i], k))
        if classify(traindata, trainlabel, valdata[i], k) == vallabel[i]:
            correction += 1
    return correction/len(vallabel)


def test(traindata, trainlabel, testdata, k, filename):
    fout = open(filename, 'w')
    for i in range(len(testdata)):
        res = classify(traindata, trainlabel, testdata, k)
        if res == 0:
            fout.write("LOW\n")
        elif res == 1:
            fout.write("MID\n")
        elif res == 2:
            fout.write("HIG\n")
        else:
            print("classify go wrong")


# 获取日志输出器
Logger = get_logger()
Eta = 0.00001
IterTimes = 250
SFold = 5
# K = 5
TrainFile = open("text_train_out_withoutend.txt")
TestFile = open("text_test_out_withoutend.txt")
Labels = get_labels("label.txt")
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
# Text = TrainText + TestText
# Sentences = [line.split(' ') for line in Text]
# Model = gensim.models.Word2Vec(Sentences, min_count=1, size=100, window=5, iter=100)
# Model.save("word_vectors_size100&window5&iter100")
Model = gensim.models.Word2Vec.load("word_vectors_size100&window5&iter100")
print("got model")
TrainX = []
ValX = []
for Sentence in TrainSentences:
    TrainX.append(get_word_vec(Model, Sentence))
for Sentence in ValSentences:
    ValX.append(get_word_vec(Model, Sentence))
TestX = []
for Sentence in TestSentences:
    TestX.append(get_word_vec(Model, Sentence))
print("got sentences")
for K in range(1, 20):
    print(str(K)+"NN:", val(TrainX, TrainLabels, ValX, ValLabels, K))
# test(TrainX, TrainLabels, TestX, K, str(K)+"NN_result.csv")
