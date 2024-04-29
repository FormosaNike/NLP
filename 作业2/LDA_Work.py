import os
import jieba
import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def extract_paragraphs(folder_path, K):
    corpus = []
    labels = []
    names = os.listdir(folder_path)
    for name in names:
        novel_name = folder_path + '\\' + name
        with open(novel_name, 'r', encoding='ANSI') as f:
            con = f.read()
            con = preprocess_text(con)
            pos = int(len(con) // 63)  #16篇文章，分词后，每篇均匀选取63个k词段落
            for i in range(63):
                corpus.append(list(con[i * pos:i * pos + K]))
                labels.append(name[:-4])
                if len(corpus) == 1000:
                    break
        f.close()
    labels = labels[:1000]
    return corpus, labels
def preprocess_text(con):
    # tokens = list(jieba.cut(con)) # 按词拆分
    tokens = list(con) # 按字拆分
    stopwords = set()  # 停用词集合，需要根据实际情况进行填充
    with open("stopwords.txt", "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip()  # 去除行末尾的换行符和空格
            stopwords.add(word)  # 将停用词添加到集合中
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [token for token in tokens if token != '\n']  # 去除换行符
    tokens = [token for token in tokens if token != '\u3000']  # 去除全角空格
    tokens = [token for token in tokens if token.strip() != '']  # 去除空格
    return tokens

# 设定参数及分类器选择
folder_path = r'E:\desktop\NLP\作业2\jyxstxtqj_downcc.com'  # 文本文件夹路径
num_token_list = [20, 100, 500, 1000, 3000]
num_topic_list = [50, 100, 200]
num_cross = 10
classifier = SVC()
results = []
# 主函数
def main():
    for K in num_token_list:
        for T in num_topic_list:
            [corpus, labels] = extract_paragraphs(folder_path, K)
            X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.1, random_state=42)

            # 将文本转换为主题分布的流水线
            lda_pipeline = Pipeline([
                ('vectorizer', CountVectorizer(max_features=K, analyzer='char')),
                ('lda', LatentDirichletAllocation(n_components=T, random_state=42, n_jobs=-1))
            ])

            # 将文本转换为主题分布
            X_train_lda = lda_pipeline.fit_transform([' '.join(x) for x in X_train])
            X_test_lda = lda_pipeline.transform([' '.join(x) for x in X_test])

            # 使用分类器进行训练和评估
            classifier.fit(X_train_lda, y_train)
            accuracy = np.mean(cross_val_score(classifier, X_train_lda, y_train, cv = num_cross))
            test_accuracy = accuracy_score(y_test, classifier.predict(X_test_lda))

            # 结果保存
            results.append(
                {
                    'K': K,
                    'T': T,
                    'Classifier': "SVM",
                    'type': 'char',
                    'training accuracy': accuracy,
                    'test accuracy': test_accuracy
                }
            )
            results_df = pd.DataFrame(results)
            results_df.to_excel("results2.xlsx", index=False)


if __name__ == '__main__':
    main()
