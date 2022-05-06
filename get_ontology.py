import pandas as pd

gwas_cat = pd.read_csv('gwas_catalog_v1.0.2-associations_e105_r2022-03-08.tsv', sep='\t')
gwas_disease_map = pd.read_csv('gwas_catalog_trait-mappings_r2022-03-08.tsv', sep='\t')


def reformat_efo(x):
    if type(x) == float:
        return "empty"
    else:
        x = str(x.split(',')[0])
        if "EFO_" in x:
            return x.rsplit('/', 1)[1]
        else:
            return "empty"


efo_link = gwas_cat[["SNPS", "MAPPED_TRAIT_URI", "CONTEXT", "DISEASE/TRAIT", "CHR_ID", "CHR_POS"]]
efo_link["MAPPED_TRAIT_URI"] = efo_link["MAPPED_TRAIT_URI"].apply(reformat_efo)
efo_link = efo_link[efo_link["MAPPED_TRAIT_URI"] != "empty"]
efo_link = efo_link.groupby(["MAPPED_TRAIT_URI"])

gwas_disease_map["EFO URI"] = gwas_disease_map["EFO URI"].apply(reformat_efo)
gwas_disease_map = gwas_disease_map[gwas_disease_map["EFO URI"] != "empty"]
gwas_disease_map_ = gwas_disease_map.groupby(["EFO URI"])

efo_to_disease = {}
for key, val in gwas_disease_map_:
    efo_to_disease[key] = val["Parent term"].unique().tolist()[0]

not_found = []
count = 0

all_frames = []
for key, val in efo_link:
    try:
        disease = efo_to_disease[key]
        is_cancer = 1 if disease.strip().lower() == 'cancer' else 0
        val['is_cancer'] = [is_cancer for _ in range(len(val))]
        val['EFO'] = [key for _ in range(len(val))]
        all_frames.append(val)
    except:
        not_found.append(key)
    count += 1

efo_merged = pd.concat(all_frames)
efo_merged.to_csv('efo_snp_category_cancer.csv', index=False)
