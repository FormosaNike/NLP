import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
import os

def preprocess_text(file_path):
    # 去除停用词和特殊字符
    corpus = ''
    r0 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open(r'E:\desktop\NLP\作业3\stopwords.txt', 'r', encoding='utf8') as f:
        stopwords = [word.strip('\n') for word in f.readlines()]
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r0, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自www.cr173.com免费txt小说下载站更多更新免费电子书请关注www.cr173.com', '')
        corpus = corpus.replace('免费小说', '')
    words = list(jieba.cut(corpus))
    return [word for word in words if word not in stopwords]

def train(dataset_path, test_name):
    """CBOW"""
    model = Word2Vec(sentences=PathLineSentences(dataset_path), hs=1, min_count=10, window=5,
                     vector_size=200, sg=0, workers=16, epochs=50)
    model.save('model1.model')
    for name in test_name:
        print(name)
        for result in model.wv.similar_by_word(name, topn=10):
            print(result[0], '{:.6f}'.format(result[1]))
        print('-------------------------')

def cluster(test_name):
    # with open('./name.txt', 'r', encoding='utf8') as f:
    #     names = f.readline().split(' ')
    model = Word2Vec.load('model1.model')
    # for name in test_name:
    #     # print(name)
    #     for result in model.wv.similar_by_word(name, topn=10):
    #         # print(result[0], '{:.6f}'.format(result[1]))
    #     # print('-------------------------')
    # names = name.readline().split(' ')
    names = [result[0] for name in test_name for result in model.wv.similar_by_word(name, topn=10)]
    name_vectors = np.array([model.wv[name] for name in names])
    # name_vectors = np.array([model.wv[result[0]] for name in test_name for result in model.wv.similar_by_word(name, topn=10)])
    tsne = TSNE()
    embedding = tsne.fit_transform(name_vectors)
    n = 6
    label = KMeans(n).fit(embedding).labels_
    plt.title('kmeans聚类结果')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(label)):
        if label[i] == 0:
            plt.plot(embedding[i][0], embedding[i][1], 'ro', )
        if label[i] == 1:
            plt.plot(embedding[i][0], embedding[i][1], 'go', )
        if label[i] == 2:
            plt.plot(embedding[i][0], embedding[i][1], 'yo', )
        if label[i] == 3:
            plt.plot(embedding[i][0], embedding[i][1], 'co', )
        if label[i] == 4:
            plt.plot(embedding[i][0], embedding[i][1], 'bo', )
        if label[i] == 5:
            plt.plot(embedding[i][0], embedding[i][1], 'mo', )
        plt.annotate(names[i], xy=(embedding[i][0], embedding[i][1]), xytext=(embedding[i][0]+0.1, embedding[i][1]+0.1))
    plt.show()
    plt.savefig('cluster.png')

folder_path = 'E:/desktop/NLP/作业3/jyxstxtqj_downcc.com'
test_name = ['李文秀', '袁承志', '胡斐', '狄云', '韦小宝', '剑客', '郭靖', '杨过'
             , '陈家洛', '萧峰', '石破天', '令狐冲', '胡斐', '张无忌', '袁冠南', '阿青']
dataset_path = 'E:/desktop/NLP/作业3/dataset'
if __name__ == '__main__':
    # with open('E:/desktop/NLP/作业3/inf.txt', 'r',  encoding='utf8') as inf:
    #     txt_list = inf.readline().split('，')
    #     for name in txt_list:
    #         file_name = name + '.txt'
    #         file_content = preprocess_text(folder_path + '/' + file_name)
    #         temp = []
    #         count = 0
    #         lines = []
    #         for w in file_content:
    #             if count % 50 == 0:
    #                 lines.append(" ".join(temp))
    #                 temp = []
    #                 count = 0
    #             temp.append(w)
    #             count += 1
    #         with open('E:/desktop/NLP/作业3/dataset/' + 'train_' + file_name, 'w', encoding='utf8') as train:
    #             train.writelines(lines)
    train(dataset_path, test_name)
    cluster(test_name)
