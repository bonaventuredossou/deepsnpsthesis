# Bonaventure Dossou - MSc Thesis (May 2022)
# Gets stats (count, frequencies) of diseases count per variant

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('efo_snp_category_cancer.csv')
dataset = dataset[["CONTEXT", "is_cancer"]]

keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']

key_info = {}


def count_disease_stats(key, list_, diseases_binary, dict_):
    key_index = [i for i in range(len(list_)) if key.strip().lower() in str(list_[i]).strip().lower()]
    disease_info_key = np.array(diseases_binary)[key_index]
    dict_[key] = [disease_info_key.tolist().count(0), disease_info_key.tolist().count(1)]


for key_ in keys:
    count_disease_stats(key_, dataset["CONTEXT"].tolist(), dataset["is_cancer"].tolist(), key_info)

X = np.arange(len(keys))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()

keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']

no_cancer_sum = sum([values[0] for key, values in key_info.items()])
cancer_sum = sum([values[1] for key_, values in key_info.items()])

no_cancer = [values[0] / no_cancer_sum for key_, values in key_info.items()]
cancer = [values[1] / cancer_sum for key_, values in key_info.items()]

rects1 = ax.bar(X + 0.00, no_cancer, width, label='no_cancer')
rects2 = ax.bar(X + 0.25, cancer, width, label='cancer')
ax.set_title(' $\%$ of Non-Cancer/Cancer Disease Per Variant')
ax.set_ylabel('Probability')
ax.set_xticks(X)
ax.set_xticklabels(keys, rotation=90)
ax.legend()
fig.tight_layout()
plt.savefig('pictures/variant_freq_is_cancer_or_not.png')
plt.show()
