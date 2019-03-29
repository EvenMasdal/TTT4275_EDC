import numpy as np
#import dataloader
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

    # Bias trick
    x = np.insert(x, x.shape[1], 1, axis=1)
    x = np.true_divide(x, np.amax(x)/2) - 1
    return x, t


def training_loop(x, t, a, iterations):
    W = np.random.normal(0, 1, (3, 5)) # 3x4 + 1 for bias

    mse_vals = []
    for i in range(iterations):
        classifications = np.dot(x, W.T)
        g = sigmoid(classifications)
        W = W - a * mse_gradient(t, g, x).T
        mse_vals.append(mse(t, g).mean())

    return W, mse_vals

def histogram(data, feature=0, _class=0, step=0.3):
    ''' Wish to make the bins of size 0.1. Max value for our data set
        (independent of feature) is 5.1 . Thus if we use the bins [1:0.1:max]
        we are sure that every feature will fall into one of the bins.
        WE COULD MAYBE USE bins = 'auto'''
    max_data_val = np.amax(data)
    slice_val = int(len(data)/3)
    bins = np.linspace(0.0 ,int(max_data_val+step), num=int((max_data_val/step)+1), endpoint=False)

    if _class == 0:
        a = data[:slice_val, feature]
    elif _class == 1:
        a = data[slice_val:2*slice_val, feature]
    else:
        a = data[2*slice_val:, feature]

    plt.hist(a, bins, alpha=0.5, label = ['hello', 'hey', 'waddup'])

def gen_histogram(data, feature, num_class=3):
    for i in range(num_class):
        histogram(data,feature,i)

if __name__ == '__main__':
    iterations = 1000
    #x, t = get_data()
    '''
    PLOT SHIT MADE BY KRIS
    data = np.genfromtxt('iris.data', dtype='U16', delimiter=',')
    x = data[:, :4].astype('float')
    y = data[:, -1]

    num_class = 3
    num_features= 4
    gen_histogram(x, feature=0)

    plt.show()

    gen_histogram(x, feature=1)
    plt.show()
    gen_histogram(x, feature=2)
    plt.show()
    gen_histogram(x, feature=3)
    plt.show()
    '''

    W, mse_vals = training_loop(x[:90], t[:90], 0.2, iterations)
    steps = list(range(iterations))

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    print(confusion_matrix(preds[:90], true[:90], 3))
    print(confusion_matrix(preds[90:], true[90:], 3))

    W, mse_vals = training_loop(x[61:], t[61:], 0.2, iterations)
    steps = list(range(iterations))

    preds = predict(W, x)
    true = np.argmax(t, axis=1)
    print('Inverse order')
    print(confusion_matrix(preds[61:], true[61:], 3))
    print(confusion_matrix(preds[:60], true[:60], 3))


    plt.plot(steps, mse_vals)
    plt.ylim(0, 0.3)
    plt.show()