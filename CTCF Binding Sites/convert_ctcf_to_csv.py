import pandas as pd
import numpy as np

ctcf_file = 'CTCFBSDB_all_exp_sites.txt'

chrs, gene_start, gene_end = [], [], []
with open(ctcf_file, encoding='utf8') as ctcf_records:
    records = ctcf_records.readlines()
    headers = records[0].split('\t')

    chromosome = headers.index('Chr')
    genome_st, genome_end = headers.index('GenomeBeginLoc'), headers.index('GenomeEndLoc')

    for line in records[1:]:
        try:
            chrs.append(int(line.split('\t')[chromosome][3:]))
            gene_start.append(float(line.split('\t')[genome_st]))
            gene_end.append(float(line.split('\t')[genome_end]))
        except:
            pass
frame = pd.DataFrame()

frame['Chr_ID'] = chrs
frame['Gene_Start'] = gene_start
frame['Gene_End'] = gene_end
distances = np.array(gene_end) + np.array(gene_start)
distances /= 2
frame['binding_site'] = distances.tolist()

frame.to_csv('ctcf_binding_sites_info.csv', index=False)
