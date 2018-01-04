import string


def read_train_file(in_filename, train_filename, label_filename):
    fin = open(in_filename, 'r')
    train_f = open(train_filename, 'w')
    label_f = open(label_filename, 'w')

    for line in fin.readlines():
        line = line.strip('\n')
        temp = line.split('\t\t')
        label_f.write(temp[0] + '\n')
        temp[1] = remove_punctuation(temp[1])
        words = temp[1].split(' ')
        for word in words:
            if word != '' and word != 'sssss':
                train_f.write(word + ' ')
        train_f.write('\n')
    fin.close()
    train_f.close()
    label_f.close()


def remove_punctuation(s):
    blanks = ""
    for i in range(32):
        blanks += " "
    table = bytes.maketrans(bytes(string.punctuation, encoding='utf-8'), bytes(blanks, encoding='utf-8'))
    s = s.translate(table)
    return s


TrainFileText = "MulLabelTrain.ss"
TestFileText = "MulLabelTest.ss"
OutTrainFile = "TrainText.csv"
OutTestFile = "TestText.csv"
OutLabels = "label.csv"
read_train_file(TrainFileText, OutTrainFile, OutLabels)
read_train_file(TestFileText, OutTestFile, "should_be_deleted")