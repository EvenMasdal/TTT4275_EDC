import numpy as np


vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


def get_data():
    data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
    identifiers = data[:, 0]
    data = data[:, 10:13].astype(np.int)
    return data, identifiers


def find_indeces(find, data):
    return np.flatnonzero(np.core.defchararray.find(data,find)!=-1)


def split_training_test(identifiers, data):
    train_idx, test_idx = [], []

    for vowel in vowels:
        idx = find_indeces(vowel, identifiers)
        train_idx.extend(idx[:70])
        test_idx.extend(idx[70:])

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    return data[train_idx], identifiers[train_idx], data[test_idx], identifiers[test_idx]


def cov_matrix_temp(x, y):
    covs = []
    for i, vowel in enumerate(vowels):
        idx = find_indeces(vowel, y)
        class_samples = x[idx]
        sample_mean = class_samples.mean(axis=0)

        subt = class_samples - sample_mean
        cov1 = np.dot(subt.T, subt)/70
        cov2 = np.cov(class_samples.T)

        covs.append(cov1)

    return covs


def main():
    data, identifiers = get_data()

    x_train, y_train, x_test, y_test = split_training_test(identifiers, data)

    a = cov_matrix_temp(x_train, y_train)
    for i in a:
        print(i)

if __name__ == '__main__':
    main()