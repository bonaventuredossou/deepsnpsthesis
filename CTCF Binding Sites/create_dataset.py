import pandas as pd
import json
from sklearn.model_selection import train_test_split

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


f = open('min_dist_snps_ctcf.json')
data = json.load(f)
variants = list(data.keys())

dataset_["CHR_ID"] = dataset_["CHR_ID"].apply(filter_ids)
dataset_["CHR_ID"] = dataset_[dataset_["CHR_ID"] != "empty"]["CHR_ID"]
group_by_variant = dataset_.groupby(by='CONTEXT')

all_frames = []
for variant, group in group_by_variant:
    if variant in keys:
        cancer_snps_chr_ids = group[group.is_cancer == 1]
        not_cancer_snps_chr_ids = group[group.is_cancer == 0]

        real_cancer_snps = list(
            set(cancer_snps_chr_ids["SNPS"].tolist()).difference(set(not_cancer_snps_chr_ids["SNPS"].tolist())))

        real_notcancer_snps = list(
            set(not_cancer_snps_chr_ids["SNPS"].tolist()).difference(set(cancer_snps_chr_ids["SNPS"].tolist())))

        cancer_snps_chr_ids = cancer_snps_chr_ids[cancer_snps_chr_ids["SNPS"].isin(real_cancer_snps)]
        cancer_snps_chr_ids['min_distance'] = data[variant][0]

        all_frames.append(cancer_snps_chr_ids)

        not_cancer_snps_chr_ids = not_cancer_snps_chr_ids[not_cancer_snps_chr_ids["SNPS"].isin(real_notcancer_snps)]
        not_cancer_snps_chr_ids['min_distance'] = data[variant][1]

        all_frames.append(not_cancer_snps_chr_ids)


def convert_chr_position_to_int(x):
    try:
        return int(x)
    except:
        return 1000


def convert_is_cancer_to_int(x):
    return int(x)


training = pd.concat(all_frames)
training['CHR_ID'] = training["CHR_ID"].apply(convert_chr_position_to_int)
training = training[training["CHR_ID"] != 1000]
training['is_cancer'] = training["is_cancer"].apply(convert_is_cancer_to_int)

SNPS_dict = {label: value + 1 for value, label in enumerate(training["SNPS"].unique().tolist())}
VARIANTS_dict = {variant: value + 1 for value, variant in enumerate(keys)}


def convert_snps_to_int(snp):
    return SNPS_dict[snp]


def convert_variants_to_int(variant_):
    return VARIANTS_dict[variant_]


training = training[["SNPS", "CONTEXT", "CHR_ID", "CHR_POS", "min_distance", "is_cancer"]]
to_filter = True

if to_filter:
    good_features = ['intron_variant', 'intergenic_variant', 'TF_binding_site_variant']
    training = training[training["CONTEXT"].isin(good_features)]

training['SNPS'] = training["SNPS"].apply(convert_snps_to_int)
training['CONTEXT'] = training["CONTEXT"].apply(convert_variants_to_int)

training = training.sample(frac=1, random_state=1234)

cancer = training[training["is_cancer"] == 1]
non_cancer = training[training["is_cancer"] == 0]
# (Non-cancer, Cancer) (284337, 7352)

non_cancer = non_cancer.sample(n=2 * len(cancer), random_state=1234)

def dataset_splits(dataframe):
    train, test = train_test_split(dataframe, test_size=0.1, random_state=1234)
    real_train, valid = train_test_split(train, test_size=0.1, random_state=1234)

    return real_train, valid, test


train_cancer, valid_cancer, test_cancer = dataset_splits(cancer)
train_non_cancer, valid_non_cancer, test_non_cancer = dataset_splits(non_cancer)

training_deep_snps = pd.concat([train_cancer, train_non_cancer])
validation_deep_snps = pd.concat([valid_cancer, valid_non_cancer])
test_deep_snps = pd.concat([test_cancer, test_non_cancer])

training_deep_snps = training_deep_snps.sample(frac=1, random_state=1234)
validation_deep_snps = validation_deep_snps.sample(frac=1, random_state=1234)
test_deep_snps = test_deep_snps.sample(frac=1, random_state=1234)

training_deep_snps.to_csv('training_data_filtered.csv', index=False)
validation_deep_snps.to_csv('validation_data_filtered.csv', index=False)
test_deep_snps.to_csv('test_data_filtered.csv', index=False)
