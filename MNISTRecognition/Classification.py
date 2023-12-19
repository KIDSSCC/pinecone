from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_classifier(arg):
    if arg == 'K-Nearest':
        return KNeighborsClassifier(n_neighbors=10)
    if arg == 'NaiveBayes':
        return MultinomialNB()
    if arg == 'DecisionTree':
        return DecisionTreeClassifier()
    if arg == 'RandomForest':
        return RandomForestClassifier()
    if arg == 'LogisticRegression':
        return LogisticRegression()
    if arg == 'SupportVectorMachine':
        return SVC()
    raise ValueError("No means matched")


def classification(train_image, train_label, test_image, test_label, classifier_item='K-Nearest'):
    print(classifier_item)
    print(len(train_image))
    print(len(train_label))

    classifier = get_classifier(classifier_item)
    classifier.fit(train_image, train_label.ravel())
    predict = classifier.predict(test_image)
    print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
    print("Classification report for classifier %s:\n%s\n" % (classifier, classification_report(test_label, predict)))
