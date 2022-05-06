import json
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import mean
from numpy import std

f = open('min_dist_snps_ctcf.json')
data = json.load(f)
variants = list(data.keys())

NAMES = ['intron', 'intergenic', 'regulatory', 'missense', 'stop_gained',
         'non_coding_exon', '3_prime_UTR', '5_prime_UTR', 'TF_binding_site']


def distribution_function(x):
    mean_ = mean(x)
    std_ = std(x)
    dist = norm(mean_, std_)
    return dist.pdf(x)


def transform_array(list_of_distances):
    return distribution_function(list_of_distances)


def plot_histogram(cancer_, not_cancer_, bins_, ax, index):
    cancer_pdf = transform_array(cancer_)
    not_cancer_pdf = transform_array(not_cancer_)
    ax.plot(cancer_, cancer_pdf, label='cancer', color='red')
    ax.plot(not_cancer_, not_cancer_pdf, alpha=0.7, color='violet', label='not_cancer')
    ax.set_xlabel('ln($min_{distances}$)')
    ax.set_ylabel('pdf')
    title = '{}, ctcf'.format(NAMES[index - 1])
    ax.set_title('dist({})'.format(title), fontsize=8)
    ax.legend(loc='best', prop={'size': 5})


for i in range(1, 10):
    cancer, not_cancer = np.array(data[variants[i - 1]][0]), np.array(data[variants[i - 1]][1])
    cancer[cancer <= 0] = 1e-2  # replace -1, 0 by 1e-2 (a small number) to avoid -inf
    not_cancer[not_cancer <= 0] = 1e-2

    cancer, not_cancer = np.log(cancer), np.log(not_cancer)
    bins = np.linspace(min(cancer), max(cancer), len(cancer))

    subax = plt.subplot(330 + i)
    plot_histogram(cancer, not_cancer, bins, subax, i)

plt.savefig('../pictures/pdf_shortest_distance_snps_ctcf_binding_site.png')
plt.show()
