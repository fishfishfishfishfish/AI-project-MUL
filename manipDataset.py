import string
import numpy
import random


class WordVector(object):
    def __init__(self):
        self.words_idf = {}
        self.word_vectors = []
        self.file_size = 0

    def get_word_vector(self, lines):
        self.file_size = len(lines)
        for words in lines:
            for word in set(words):
                if word not in self.words_idf:
                    self.words_idf[word] = 1
                    self.word_vectors.append(word)
                else:
                    self.words_idf[word] += 1
        self.normalize_words_idf()

    def normalize_words_idf(self):
        idf_list = []
        for k, v in self.words_idf.items():
            self.words_idf[k] = self.file_size / (1 + v)
            idf_list.append(self.words_idf[k])
        idf_list = numpy.array(idf_list)
        # minimal = idf_list.min()
        # maximal = idf_list.max()
        m = idf_list.mean()
        s = idf_list.std()
        for k, v in self.words_idf.items():
            # self.words_idf[k] = (v-minimal)/(maximal-minimal)
            # self.words_idf[k] = 100 * (v-m)/s  # std = 100, mean = 0
            self.words_idf[k] = 50 * (v-m)/s + 100  # std = 100, mean = 200


    def split_word_vector(self, split_amount):
        random.seed(1)
        temp_vec = self.word_vectors.copy()
        self.word_vectors = []
        split_size = len(temp_vec)//split_amount
        for i in range(split_amount-1):
            split_vec = []
            for j in range(split_size):
                split_vec.append(temp_vec.pop(random.randint(0, len(temp_vec)-1)))
            self.word_vectors.append(split_vec)
        self.word_vectors.append(temp_vec)

    def get_most_word_vec(self, how_many):
        res_word_vec = []
        dictionary = sorted(self.words_idf.items(), key=lambda d: d[1], reverse=False)
        for di in range(how_many):
            res_word_vec.append(dictionary[di][0])
        return res_word_vec

    def cal_tfidf(self, line, word_vec):
        # word_vec = self.word_vectors[word_vec_index]
        line_tfidf = [0 for t in range(len(word_vec))]
        for word in word_vec:
            if len(line) != 0:
                line_tfidf[word_vec.index(word)] = (line.count(word) / len(line) * self.words_idf[word])
        return line_tfidf


def read_text(filename):
    f = open(filename, 'r')
    lines = []
    for line in f.readlines():
        line = line.strip('\n')
        words = line.split(' ')
        while '' in words:
            words.remove('')
        lines.append(words)
    return lines


def write_file(filename, wv, lines, word_vec):
    outfile = open(filename, 'w')
    # cnt = 0
    # word_vec = wv.word_vectors[word_vec_index].copy()
    outfile.write(word_vec[0])
    for w in range(1, len(word_vec)):
        outfile.write(',' + word_vec[w])
    outfile.write('\n')
    for line in lines:
        # cnt += 1
        # if cnt % 100 == 0:
        #     print(cnt)
        line_tfidf = wv.cal_tfidf(line, word_vec)
        outfile.write(str(line_tfidf[0]))
        for t in range(1, len(line_tfidf)):
            outfile.write(',' + str(line_tfidf[t]))
        outfile.write('\n')
    outfile.close()


TrainTextFileName = "text_train_out_withoutend.txt"
TestTextFileName = "text_test_out_withoutend.txt"
# TrainIDFFileName = "Train_TFIDF_dense.csv"
# TestIDFFileName = "Test_TFIDF_dense.csv"
TrainLines = read_text(TrainTextFileName)
TestLines = read_text(TestTextFileName)
WV = WordVector()
WV.get_word_vector(TrainLines)
Dictionary = sorted(WV.words_idf.items(), key=lambda d: d[1], reverse=False)
IDFFileName = "IDF_dic.txt"
IDFFile = open(IDFFileName, 'w')
for Di in Dictionary:
    IDFFile.write(str(Di)+'\n')
IDFFile.close()
print("got idf")
# WV.split_word_vector(40)
WordVec = WV.get_most_word_vec(4000)
TrainIDFFileName = "Train_TFIDF_dense.csv"
TestIDFFileName = "Test_TFIDF_dense.csv"
write_file(TrainIDFFileName, WV, TrainLines, WordVec)
print("got Train")
write_file(TestIDFFileName, WV, TestLines, WordVec)
print("got Test")
# for FileIndex in range(0, 3):
#     TrainIDFFileName = "Train_TFIDF_dense_" + str(FileIndex) + ".csv"
#     TestIDFFileName = "Test_TFIDF_dense_" + str(FileIndex) + ".csv"
#     WordVec = WV.word_vectors[FileIndex]
#     write_file(TrainIDFFileName, WV, TrainLines, WordVec)
#     print("got Train", FileIndex)
#     write_file(TestIDFFileName, WV, TestLines, WordVec)
#     print("got Test", FileIndex)
