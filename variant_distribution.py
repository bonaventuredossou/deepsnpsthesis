import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_ = pd.read_csv('efo_snp_category_cancer.csv')
dataset_ = dataset_[["SNPS", "CONTEXT", "is_cancer", "DISEASE/TRAIT"]]

NAMES = ['intron', 'intergenic', 'regulatory', 'missense', 'stop_gained',
         'non_coding_exon', '3_prime_UTR', '5_prime_UTR', 'TF_binding_site']


def reformat_snps(snp):
    snp = snp.split(";")
    snp = [idx.strip() for idx in snp if idx.startswith("rs")]
    try:
        snp = str(snp[0])
    except:
        snp = 'empty'
    return snp


dataset_ = dataset_.drop_duplicates(subset=["SNPS"], inplace=False)
dataset_['SNPS'] = dataset_['SNPS'].apply(reformat_snps)
dataset_ = dataset_[dataset_["SNPS"] != "empty"]

keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']

groups = dataset_.groupby(by="CONTEXT")
variant_count = {}
for key, group in groups:
    if key in keys:
        variant_count[key] = len(group)


def plot_variant_count(count_dict):
    variants = list(count_dict.keys())
    counts = list(count_dict.values())
    plt.bar(np.arange(len(variants)) + 0.00, counts, 0.25,
            color=['blue', 'orange', 'brown', 'green', 'pink', 'red', 'violet', 'gold', 'black'])
    plt.ylabel('frequency')
    plt.xlabel('Variant')
    plt.xticks(np.arange(len(variants)), labels=variants, rotation=90)
    plt.savefig('pictures/variants_frequency.png')
    plt.show()


def most_common_snp():
    if not os.path.exists('snps_count.csv'):
        snps = dataset_.SNPS.tolist()
        unique_snps = dataset_.SNPS.unique().tolist()
        snps_count = {}
        for snp in unique_snps:
            snps_count[snp] = snps.count(snp)
        snps_count_frame = pd.DataFrame()
        snps_count_frame['SNPS'] = list(snps_count.keys())
        snps_count_frame['count'] = list(snps_count.values())
        snps_count_frame.sort_values(by='count', inplace=True, ascending=False)
        snps_count_frame = snps_count_frame[:30]
        snps_count_frame.to_csv('snps_count.csv', index=False)

    else:
        snps_count_ = pd.read_csv('snps_count.csv')
        snps_count_frame = pd.DataFrame(index=snps_count_['SNPS'])
        snps_count_frame['count'] = snps_count_["count"].tolist()

    snps_count_frame.plot.bar()
    plt.xlabel('SNPs')
    plt.ylabel('frequency')
    plt.savefig('pictures/snps_frequency.png')
    plt.show()


most_common_snp()


def get_type_disease_per_snp():
    top_30 = pd.read_csv('snps_count.csv')
    top_30 = top_30.SNPS.tolist()
    group_snps = dataset_.groupby(by='SNPS')
    snps_dict = {}
    for snp, group_ in group_snps:
        if snp in top_30:
            cancer_ = len(group_[group_["is_cancer"] == 1])
            not_cancer_ = len(group_[group_["is_cancer"] == 0])
            snps_dict[snp] = (cancer_, not_cancer_)

    X = np.arange(len(snps_dict))
    width = 0.25
    fig, ax = plt.subplots()

    no_cancer_info = [value[1] for key_, value in snps_dict.items()]
    is_cancer_info = [value[0] for key_, value in snps_dict.items()]

    rects1 = ax.bar(X + 0.00, no_cancer_info, width, label='no_cancer')
    rects2 = ax.bar(X + 0.25, is_cancer_info, width, label='cancer')

    plt.ylabel('frequency (non_cancer, cancer)')
    plt.xlabel('SNPs')
    plt.xticks(X, list(snps_dict.keys()), rotation=90)
    plt.legend()
    fig.tight_layout()
    plt.savefig('pictures/disease_tupe_snp_per.png')
    plt.show()

get_type_disease_per_snp()