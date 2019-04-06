import numpy as np
import matplotlib.pyplot as plt

# One hot encode data
classmapping = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}


def mse(A, B):
    return ((A-B)**2).mean(axis=1)


def mse_gradient(t, g, x):
    mse_grad = g-t
    g_grad = g * (1-g) # Element wise
    W_grad = x.T

    return np.dot(W_grad, mse_grad*g_grad)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(W, x):
    x = sigmoid(np.dot(x, W.T))
    if x.ndim == 1:
        return np.argmax(x)
    else:
        return np.argmax(x, axis=1)


def confusion_matrix(pred, true, num_classes):
    conf = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        conf[true[i]][pred[i]] += 1
    return conf


def get_data():
    x = []
    t = []
    with open('iris.data', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if line == '':
                continue
            splt = line.split(',')
            properties = [float(i) for i in splt[:-1]]
            classification = splt[-1]
            x.append(np.array(properties))
            t.append(classmapping.get(classification,''))

    proper_x = []
    proper_t = []
    for i in range(50):
        for j in range(3):
            proper_x.append(x[50*j+i])
            proper_t.append(t[50*j+i])

    x = np.array(proper_x)
    t = np.array(proper_t)

    x = np.true_divide(x, np.amax(x)/2) - 1
    # Bias trick
    x = np.insert(x, x.shape[1], 1, axis=1)
    return x, t


def training_loop(x, t, a, iterations):
    W = np.random.normal(0, 1, (3, x.shape[1]))

    mse_vals = []
    for i in range(iterations):
        classifications = np.dot(x, W.T)
        g = sigmoid(classifications)
        W = W - a * mse_gradient(t, g, x).T
        mse_vals.append(mse(t, g).mean())

    return W, mse_vals


def print_confusion(conf):

    classes = ['1', '2', '3']


    print("""\\begin{table}[H]
\\caption{}
\\centering
\\begin{tabular}{|c|lll|}""")
    conf = conf.astype(int)
    print('\\hline\nclass & '+' & '.join(classes) + '\\\\' + '\\hline')
    for i, row in enumerate(conf):
        rw = classes[i]
        for j, elem in enumerate(row):
            rw += ' & '
            if elem == 0:
                rw += '-'
            else:
                rw += str(elem)
        rw += '\\\\'
        if i == 2:
            rw += '\\hline'
        print(rw)
    print("""\\end{tabular}
\\end{table}""")
    print()

if __name__ == '__main__':
    iterations = 20
    x, t = get_data()

    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

#    print_confusion(confusion_matrix(preds[:90], true[:90], 3))
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))

    # Remove feature 2
    x = np.delete(x, 1, axis=1)
    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

#    print_confusion(confusion_matrix(preds[:90], true[:90], 3))
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))

    # Remove feature 1
    x = np.delete(x, 0, axis=1)
    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

#    print_confusion(confusion_matrix(preds[:90], true[:90], 3))
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))

    # Remove feature 4
    x = np.delete(x, 1, axis=1)
    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)


    preds = predict(W, x)
    true = np.argmax(t, axis=1)

#    print_confusion(confusion_matrix(preds[:90], true[:90], 3))
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))