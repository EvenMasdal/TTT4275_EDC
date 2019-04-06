import numpy as np
import matplotlib.pyplot as plt


def gen_histogram(data, feature, plt_axis, step=0.3):
    max_val = np.amax(data)         # Finds maxvalue in samples
    slice_val = int(len(data)/3)    # slice variables used for slicing samples by class
    # Create bins (sizes of histogram boxes)
    bins = np.linspace(0.0 ,int(max_val+step), num=int((max_val/step)+1), endpoint=False)

    legends = ['Class 1: Setosa', 'Class 2: Versicolour', 'Class 3: Virginica']
    colors = ['Red', 'Blue', 'lime']
    features = {0: 'sepal length',
                1: 'sepal width',
                2: 'petal length',
                3: 'petal width'}

    # Slices samples by class
    samples = [data[:slice_val, feature], data[slice_val:2*slice_val, feature], data[2*slice_val:, feature]]

    # Creates plot shit, legends and subtitles
    for i in range(3):
        plt_axis.hist(samples[i], bins, alpha=0.5, stacked=True, label=legends[i], color=colors[i])
    plt_axis.legend(prop={'size': 7})
    plt_axis.set_title(f'feature {feature+1}: {features[feature]}')


if __name__ == '__main__':    
    data = np.genfromtxt('iris.data', dtype='U16', delimiter=',')
    x = data[:, :4].astype('float')
    y = data[:, -1]

    # Subplotting of histogram with shared x- and y-axis
    f, axis = plt.subplots(2,2, sharex='col', sharey='row')
    features = [0,1,2,3]
    f.suptitle('Feature histograms')
    gen_histogram(x, features[0], axis[0,0])
    gen_histogram(x, features[1], axis[0,1])
    gen_histogram(x, features[2], axis[1,0])
    gen_histogram(x, features[3], axis[1,1])
    # Adding labels to axis
    for ax in axis.flat:
        ax.set(xlabel='Measure [cm]', ylabel='Number of samples')
        ax.label_outer() # Used to share labels on y-axis and x-axis
    plt.show()
