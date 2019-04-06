import numpy as np
import scipy
import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


def get_data():
    data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
    identifiers = data[:, 0]
    data = data[:, 10:13].astype(np.int)
    return data, identifiers


def find_indeces(find, data):
    return np.flatnonzero(np.core.defchararray.find(data,find)!=-1) # Magic


def split_training_test(identifiers, data):
    train_idx, test_idx = [], []

    for vowel in vowels:
        idx = find_indeces(vowel, identifiers)
        train_idx.extend(idx[:70])
        test_idx.extend(idx[70:])

    return data[train_idx], identifiers[train_idx], data[test_idx], identifiers[test_idx]


def create_some_gmm_classifier(x, y, x_test, y_test, n_components):

    training_preds = np.zeros((12, x.shape[0]))
    testing_preds = np.zeros((12, x_test.shape[0]))

    #print(y_test)

    for i, vowel in enumerate(vowels):
        # Filter classes
        idx = find_indeces(vowel, y)
        class_samples = x[idx]

        gmm = GMM(n_components=n_components, covariance_type='diag', reg_covar=1e-4, random_state=0)

        gmm.fit(class_samples, y[idx])

        rvs = []
        for j in range(n_components):
            N = multivariate_normal(mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
            training_preds[i] += gmm.weights_[j] * N.pdf(x)
            testing_preds[i] += gmm.weights_[j] * N.pdf(x_test)

    predicted_train = np.argmax(training_preds, axis=0)
    predicted_test = np.argmax(testing_preds, axis=0)

    true_train = np.asarray([i for i in range(12) for _ in range(70)])
    true_test = np.asarray([i for i in range(12) for _ in range(69)])

    return (confusion_matrix(predicted_train, true_train, 12), np.sum(predicted_train==true_train)/len(predicted_train),
            confusion_matrix(predicted_test, true_test, 12), np.sum(predicted_test==true_test)/len(predicted_test))


def confusion_matrix(pred, true, num_classes):
    conf = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        conf[true[i]][pred[i]] += 1
    return conf


def print_confusion(conf):
    conf = conf.astype(int)
    print('\\hline\nclass & '+' & '.join(vowels) + '\\\\' + '\\hline')
    for i, row in enumerate(conf):
        rw = vowels[i]
        for j, elem in enumerate(row):
            rw += ' & '
            if elem == 0:
                rw += '-'
            else:
                rw += str(elem)
        rw += '\\\\'
        if i == 11:
            rw += '\\hline'
        print(rw)
    print()


def main():
    data, identifiers = get_data()
    x_train, y_train, x_test, y_test = split_training_test(identifiers, data)

    a = create_some_gmm_classifier(x_train, y_train, x_test, y_test, 2)
    print_confusion(a[0])
    print_confusion(a[2])
    print('===')
    print(100*(1-a[1]))
    print(100*(1-a[3]))


if __name__ == '__main__':
    main()
