import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ctcf_dataset = pd.read_csv('ctcf_binding_sites_info.csv')
efo_dataset = pd.read_csv('../efo_snp_category_cancer.csv')
chromosome_cancer = efo_dataset[["CHR_ID", "is_cancer"]]


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


chromosome_cancer["CHR_ID"] = chromosome_cancer["CHR_ID"].apply(filter_ids)
chromosome_cancer["CHR_ID"] = chromosome_cancer[chromosome_cancer["CHR_ID"] != "empty"]["CHR_ID"]

chr_ids = chromosome_cancer["CHR_ID"].unique().tolist()

chrs_dict = {}
for chromosome in chr_ids:
    try:

        if '-' in chromosome:
            chr1, chr2 = chromosome.split('-')
            chr1, chr2 = int(chr1), int(chr2)
            grouped = chromosome_cancer[chromosome_cancer["CHR_ID"] == chromosome]
            cancer = grouped[grouped["is_cancer"] == 1]
            not_cancer = grouped[grouped["is_cancer"] == 0]

            if chr1 in list(chrs_dict.keys()):
                chrs_dict[chr1] = (chrs_dict[chr1][0] + len(cancer), chrs_dict[chr1][1] + len(not_cancer))
            else:
                chrs_dict[chr1] = (len(cancer), len(not_cancer))

            if chr2 in list(chr_ids.keys()):
                chrs_dict[chr2] = (chrs_dict[chr2][0] + len(cancer), chrs_dict[chr2][1] + len(not_cancer))
            else:
                chrs_dict[chr2] = (len(cancer), len(not_cancer))
        else:
            grouped = chromosome_cancer[chromosome_cancer["CHR_ID"] == chromosome]
            cancer = grouped[grouped["is_cancer"] == 1]
            not_cancer = grouped[grouped["is_cancer"] == 0]

            if int(chromosome) in list(chrs_dict.keys()):
                chrs_dict[int(chromosome)] = (
                    chrs_dict[int(chromosome)][0] + len(cancer), chrs_dict[int(chromosome)][1] + len(not_cancer))
            else:
                chrs_dict[int(chromosome)] = (len(cancer), len(not_cancer))
    except:
        pass

cancer = [value[0] for key, value in chrs_dict.items()]
non_cancer = [value[1] for key, value in chrs_dict.items()]
chrs_ids = list(chrs_dict.keys())

X = np.arange(len(chrs_ids))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()

rects1 = ax.bar(X + 0.00, non_cancer, width, label='no_cancer')
rects2 = ax.bar(X + 0.25, cancer, width, label='cancer')
ax.set_title('Count of Cancer/Non-Cancer Disease per Chromosome')
ax.set_ylabel('Frequency Count')
ax.set_xticks(X)
ax.set_xlabel('Chromosome')
ax.set_xticklabels(chrs_ids, rotation=90)
ax.legend()
fig.tight_layout()
plt.savefig('../pictures/chromosomes_freq_is_cancer_or_not.png')
plt.show()