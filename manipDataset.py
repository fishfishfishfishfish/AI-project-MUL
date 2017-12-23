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
        for k, v in self.words_idf.items():
            self.words_idf[k] = self.file_size / (1 + v)

    def split_word_vector(self, split_amount):
        temp_vec = self.word_vectors.copy()
        self.word_vectors = []
        split_size = len(temp_vec)//split_amount
        for i in range(split_amount-1):
            split_vec = []
            for j in range(split_size):
                split_vec.append(temp_vec.pop(random.randint(0, len(temp_vec)-1)))
            self.word_vectors.append(split_vec)
        self.word_vectors.append(temp_vec)

    def cal_tfidf(self, line, word_vec_index):
        word_vec = self.word_vectors[word_vec_index]
        line_tfidf = [0 for t in range(len(word_vec))]
        for word in word_vec:
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


def write_file(filename, wv, lines, word_vec_index):
    outfile = open(filename, 'w')
    cnt = 0
    word_vec = wv.word_vectors[word_vec_index]
    for w in word_vec:
        outfile.write(w+' ')
    outfile.write('\n')
    for line in lines:
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
        line_tfidf = wv.cal_tfidf(line, word_vec_index)
        for t in line_tfidf:
            outfile.write(str(t)+',')
        outfile.write('\n')


TrainTextFileName = "text_train_out_withoutend.txt"
TestTextFileName = "text_test_out_withoutend.txt"
TrainIDFFileName = "Train_TFIDF.csv"
TestIDFFileName = "Test_TFIDF.csv"
Lines = read_text(TrainTextFileName)
WV = WordVector()
WV.get_word_vector(Lines)
WV.split_word_vector(20)
write_file(TrainIDFFileName, WV, Lines, 0)
print("got Train")
Lines = read_text(TestTextFileName)
write_file(TestIDFFileName, WV, Lines, 0)
