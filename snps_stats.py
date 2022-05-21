# Bonaventure Dossou - MSc Thesis (May 2022)
# Get Stats about SNPs anf their preferences to variants/diseases

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('efo_snp_category_cancer.csv')
dataset = dataset[["SNPS", "CONTEXT", "is_cancer"]]


def reformat_snps(snp):
    snp = snp.split(";")
    snp = [idx.strip() for idx in snp if idx.startswith("rs")]
    try:
        snp = str(snp[0])
    except:
        snp = 'empty'
    return snp


initial_len = len(dataset)
dataset = dataset.drop_duplicates(subset=["SNPS"], inplace=False)

dataset['SNPS'] = dataset['SNPS'].apply(reformat_snps)
dataset = dataset[dataset["SNPS"] != "empty"]

keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']

NAMES = ['intron', 'intergenic', 'regulatory', 'missense', 'stop_gained',
         'non_coding_exon', '3_prime_UTR', '5_prime_UTR', 'TF_binding_site']


def get_snp_cancer_information(dataset_):
    no_cancer_var = []
    cancer_var = []
    for key in keys:
        key_dataset = dataset_[dataset_["CONTEXT"] == key]
        cancer_information = key_dataset.groupby(by="SNPS")
        cancer = []
        no_cancer = []
        for snp, group in cancer_information:
            snp_info = group.groupby(by='is_cancer')
            snp_cancer_info = group[group["is_cancer"] == 1]
            snp_no_cancer_info = group[group["is_cancer"] == 0]

            snps_cancer = set(snp_cancer_info.SNPS.tolist())
            snps_no_cancer = set(snp_no_cancer_info.SNPS.tolist())

            # we don't want SNPs linked both to cancer and non-cancer diseases
            if len(snps_cancer.intersection(snps_no_cancer)) == 0:
                for is_cancer, info_group in snp_info:
                    if is_cancer == 0:
                        no_cancer.append(len(info_group))
                    else:
                        cancer.append(len(info_group))
        no_cancer_var.append(sum(no_cancer))
        cancer_var.append(sum(cancer))
    return no_cancer_var, cancer_var


no_cancer_, cancer_ = get_snp_cancer_information(dataset)


def draw_stats(no_cancer_info, is_cancer_info, title, cleaned=False):
    X = np.arange(len(keys))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()

    no_cancer_info = np.array(no_cancer_info) / sum(no_cancer_info)
    is_cancer_info = np.array(is_cancer_info) / sum(is_cancer_info)

    ax.bar(X + 0.00, no_cancer_info, width, label='no_cancer')
    ax.bar(X + 0.25, is_cancer_info, width, label='cancer')

    if cleaned:
        ax.set_title('Probability Count of SNPs (After Duplicates removal)')
    else:
        ax.set_title('Probability Count of SNPs')

    ax.set_ylabel('Probability')
    ax.set_xlabel('Variant')
    ax.set_xticks(X)
    ax.set_xticklabels(NAMES, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig('pictures/{}_is_cancer_or_not_per_variant.png'.format(title))
    plt.show()


draw_stats(no_cancer_, cancer_, 'snp_count')
