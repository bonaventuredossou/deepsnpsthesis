# Bonaventure Dossou -  MSc Thesis (May 2022)
# Computes the distance to the closest CTCF binding site

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

dataset_ = pd.read_csv('../efo_snp_category_cancer.csv')
dataset_ = dataset_[["SNPS", "CONTEXT", "is_cancer", "DISEASE/TRAIT", "CHR_ID", "CHR_POS"]]

keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']


def filter_ids(x):
    if str(x).lower() not in ["nan", "x", "y", None]:
        if str(x).isdigit():
            return str(x)
        else:
            if ";" in str(x):
                list_ = list(set([i.strip() for i in list(set(str(x).split(";")))]))
                return '-'.join(list_)
            else:
                if "x" in str(x):
                    list_ = list(set([i.strip() for i in list(set(str(x).split("x")))]))
                    return '-'.join(list_)
    else:
        return "empty"


def remove_na(x):
    return x if not pd.isna(x) else "empty"


dataset_["CHR_ID"] = dataset_["CHR_ID"].apply(filter_ids)
dataset_["CHR_ID"] = dataset_[dataset_["CHR_ID"] != "empty"]["CHR_ID"]
group_by_variant = dataset_.groupby(by='CONTEXT')
BINDING_SITES = pd.read_csv('ctcf_binding_sites_info.csv')


def get_min_distance(chr_information):
    chr_information = chr_information.groupby(by='SNPS')  # this is to avoid duplicates among SNPS
    distances = []
    for snp, group_snps in chr_information:
        # check that the SNP is linked only to a unique type of disease: cancer or non-cancer
        chr_positions = group_snps["CHR_POS"].tolist()
        chr_ids = group_snps["CHR_ID"].tolist()
        for chr_pos, chr_id in zip(chr_positions, chr_ids):
            current_chr_distances = []
            try:
                _chr_ids = chr_id.split('-')
                current_binding_sites = []
                for id_ in _chr_ids:
                    current_binding_sites_per_id = BINDING_SITES[BINDING_SITES["Chr_ID"] == int(id_)]
                    current_binding_sites += current_binding_sites_per_id["binding_site"].tolist()
                for binding in current_binding_sites:
                    cur_dist = np.abs(float(chr_pos) - binding)
                    current_chr_distances.append(cur_dist)
            except:
                pass
            if len(current_chr_distances) > 0:
                distances.append(min(current_chr_distances))
            else:
                distances.append(-1)  # corresponding binding sites not found
    return distances


dict_ = {}
dict_disease = {}
for variant, group in group_by_variant:
    if variant in keys:
        cancer_snps_chr_ids = group[group.is_cancer == 1]
        not_cancer_snps_chr_ids = group[group.is_cancer == 0]

        real_cancer_snps = list(
            set(cancer_snps_chr_ids["SNPS"].tolist()).difference(set(not_cancer_snps_chr_ids["SNPS"].tolist())))

        real_notcancer_snps = list(
            set(not_cancer_snps_chr_ids["SNPS"].tolist()).difference(set(cancer_snps_chr_ids["SNPS"].tolist())))

        cancer_snps_chr_ids = cancer_snps_chr_ids[cancer_snps_chr_ids["SNPS"].isin(real_cancer_snps)]
        not_cancer_snps_chr_ids = not_cancer_snps_chr_ids[not_cancer_snps_chr_ids["SNPS"].isin(real_notcancer_snps)]

        dict_disease[variant] = (len(not_cancer_snps_chr_ids), len(cancer_snps_chr_ids))

        # gives an empty list as expected and desired - list of SNPs are distincts
        # print(list(
        # set(not_cancer_snps_chr_ids["SNPS"].tolist()).intersection(set(cancer_snps_chr_ids["SNPS"].tolist()))))

        dist_cancer = get_min_distance(cancer_snps_chr_ids)
        dist_not_cancer = get_min_distance(not_cancer_snps_chr_ids)
        dict_[variant] = (dist_cancer, dist_not_cancer)

with open("min_dist_snps_ctcf.json", 'w') as outfile:
    json.dump(dict_, outfile)

X = np.arange(len(keys))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()

keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']

no_cancer_sum = sum([values[0] for key, values in dict_disease.items()])
cancer_sum = sum([values[1] for key_, values in dict_disease.items()])

no_cancer = [values[0] / no_cancer_sum for key_, values in dict_disease.items()]
cancer = [values[1] / cancer_sum for key_, values in dict_disease.items()]

rects1 = ax.bar(X + 0.00, no_cancer, width, label='no_cancer')
rects2 = ax.bar(X + 0.25, cancer, width, label='cancer')
ax.set_ylabel('Probability')
ax.set_xticks(X)
ax.set_xticklabels(list(dict_disease.keys()), rotation=90)
ax.legend()
fig.tight_layout()
plt.savefig('../pictures/variant_freq_is_cancer_or_not_independent_sets.png')
plt.show()
