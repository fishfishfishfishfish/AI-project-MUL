import string
import numpy


class WordVector(object):
    def __init__(self, word_filter):
        self.word_filter = word_filter
        self.word_vector = []
        self.words_idf = {}
        self.file_size = 0
        self.file_lines = []

    def get_word_vector(self, lines):
        self.file_size = len(lines)
        for words in lines:
            for word in set(words):
                if word not in self.word_filter and word != '':
                    if word not in self.words_idf:
                        self.words_idf[word] = 1
                        self.word_vector.append(word)
                    else:
                        self.words_idf[word] += 1
                else:
                    while word in words:
                        words.remove(word)
            self.file_lines.append(words)
        for k, v in self.words_idf.items():
            self.words_idf[k] = self.file_size / (1 + v)

    def cal_tfidf(self, line):
        line_tfidf = [0 for t in range(len(self.word_vector))]
        for word in set(line):
            line_tfidf[self.word_vector.index(word)] = (line.count(word) / len(line) * self.words_idf[word])
        return line_tfidf


def read_train_file(filename):
    f = open(filename, 'r')
    labels = []
    lines = []
    for line in f.readlines():
        line = line.strip('\n')
        temp = line.split('\t\t')
        temp[1] = remove_punctuation(temp[1])
        words = temp[1].split(' ')
        while '' in words:
            words.remove('')
        lines.append(words)
        if temp[0] == "LOW":
            labels.append("1,0,0")
        elif temp[0] == "MID":
            labels.append("0,1,0")
        elif temp[0] == "HIG":
            labels.append("0,0,1")
        else:
            print("label go wrong!")
    return lines, labels

def write_train_file(lines, labels, wv):



def remove_punctuation(s):
    blanks = ""
    for i in range(32):
        blanks += " "
    table = bytes.maketrans(bytes(string.punctuation, encoding='utf-8'), bytes(blanks, encoding='utf-8'))
    s = s.translate(table)
    return s


FileName = "MulLabelTrain.ss"
OutFileName = "result.csv"
OutFile = open(OutFileName, 'w')
Lines, Labels = read_train_file(FileName)
WV = WordVector(['sssss'])
WV.get_word_vector(Lines)
print("got word vector")
Mat = WV.cal_tfidf()
print("got matrix")
for i in WV.word_vector:
    OutFile.write(i+',')
for i in range(len(Labels)):
    OutFile.write(Labels[i] + ' ')
    for n in Mat[i]:
        OutFile.write(str(n)+',')
    OutFile.write('\n')
