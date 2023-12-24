import random
import jieba
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
import re
import string
from sklearn.metrics import accuracy_score,classification_report


def text_to_words(file_path):
    sentences_arr = []
    lab_arr = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            lab_arr.append(line.split('_!_')[1])
            sentence = line.split('_!_')[-1].strip()
            sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）《》：]+", "", sentence)
            sentence = jieba.lcut(sentence, cut_all=False)
            sentences_arr.append(sentence)
    return sentences_arr, lab_arr


def load_stopwords(file_path):
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords


def get_dict(sentences_arr, stopwords):
    word_dic = {}
    for sentence in sentences_arr:
        for word in sentence:
            if word != '' and word.isalpha():
                if word not in stopwords:
                    word_dic[word] = word_dic.get(word, 1) + 1
    word_dic = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)
    return word_dic


def get_feature_word(word_dic, word_num):
    n = 0
    feature_word = []
    for word in word_dic:
        if n < word_num:
            feature_word.append(word[0])
        n += 1
    return feature_word


def get_text_features(train_data_list, test_data_list, feature_words):
    def text_features(text, features_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in features_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def load_sentence(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）《》：]+", "", sentence)  # 去除标点符号
    sentence = jieba.lcut(sentence, cut_all=False)
    return sentence


if __name__ == "__main__":
    sentence_arr, lab_arr = text_to_words('data/data6826/news_classify_data.txt')
    stopwords = load_stopwords('data/data43470/stopwords_cn.txt')
    word_dic = get_dict(sentence_arr, stopwords)
    train_data_list, test_data_list, train_class_list, test_class_list = model_selection.train_test_split(sentence_arr,
                                                                                                          lab_arr,
                                                                                                          test_size=0.1)
    feature_words = get_feature_word(word_dic, 10000)
    train_feature_list, test_feature_list = get_text_features(train_data_list, test_data_list, feature_words)
    # 开始进行训练
    print('begin to train')
    classifier = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
    classifier.fit(train_feature_list, train_class_list)
    predict = classifier.predict(test_feature_list)
    test_accuracy = accuracy_score(predict, test_class_list)
    print("accuracy_score: %.4lf" % test_accuracy)
    print("Classification report for classifier:\n", classification_report(test_class_list, predict))

    lab = ['文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '国际', '证券']
    p_data = '【中国稳健前行】应对风险挑战必须发挥制度优势'
    sentence = load_sentence(p_data)
    sentence = [sentence]
    print('分词结果:', sentence)
    # 形成特征向量
    p_words = get_text_features(sentence, sentence, feature_words)
    res = classifier.predict(p_words[0])
    print("所属类型：", lab[int(res)])
