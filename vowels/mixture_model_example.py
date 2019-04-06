import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

size = 600

def plot_normal(ax, data, color, scaling=1):
    mu = data.mean()
    variance = np.var(data)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, scaling*stats.norm.pdf(x, mu, sigma), color=color)


five = np.random.normal(loc=-3, scale=0.75, size=size)
six = np.random.normal(loc=-1, scale=0.75, size=size)
seven = np.random.normal(loc=1, scale=0.75, size=size)
eight = np.random.normal(loc=3, scale=0.75, size=size)


x_axis = [0.2]*size

ax2 = plt.subplot(2,1,1)
ax1 = plt.subplot(2,1,2)

for ax in [ax1, ax2]:
    ax.hist(five, alpha=0.35, stacked=True, density=True, color='b')
    ax.hist(six, alpha=0.35, stacked=True, density=True, color='r')
    ax.hist(seven, alpha=0.35, stacked=True, density=True, color='b')
    ax.hist(eight, alpha=0.35, stacked=True, density=True, color='r')

    ax.set_xlim(-6, 6)

plot_normal(ax1, five, 'b')
plot_normal(ax1, six, 'r')
plot_normal(ax1, seven, 'b')
plot_normal(ax1, eight, 'r')


plot_normal(ax2, np.concatenate((five, seven)), 'b', scaling=3)
plot_normal(ax2, np.concatenate((six, eight)), 'r', scaling=3)

plt.show()