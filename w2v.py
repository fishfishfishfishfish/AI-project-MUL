import gensim
import numpy

# TrainFile = open("text_train_out_withoutend.txt")
# TestFile = open("text_test_out_withoutend.txt")
# TrainText = TrainFile.readlines()
# TestText = TestFile.readlines()
# Text = TrainText + TestText
# Sentences = [line.split(' ') for line in Text]
# Model = gensim.models.Word2Vec(Sentences, min_count=1, size=100, window=5, iter=100)
Model = gensim.models.Word2Vec.load("word_vectors")
print(Model.most_similar(u"cheese", topn=9))
Model.save("word_vectors")
