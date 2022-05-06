import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

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

def draw_graph(key, dataset, ax, index, threshold, style):
    dataset = dataset[dataset.CONTEXT == key]
    snp_cancer_info = dataset.groupby(by="DISEASE/TRAIT")
    dict_ = {}
    for ind, (group, snp_info) in enumerate(snp_cancer_info):
        if style == "cancer_non_cancer":
            # Cancer/Non-Cancer
            if snp_info["is_cancer"].unique().tolist()[0] == 0:
                name = 'not_cancer_{}'.format(ind)
                dict_[name] = list(set(snp_info["SNPS"].tolist()))
            else:
                name = 'cancer_{}'.format(ind)
                dict_[name] = list(set(snp_info["SNPS"].tolist()))

        if style == "cancer":
            # Cancer/Cancer
            if snp_info["is_cancer"].unique().tolist()[0] == 1:
                name = 'cancer_{}'.format(ind)
                dict_[name] = list(set(snp_info["SNPS"].tolist()))

        if style == "no_cancer":
            if snp_info["is_cancer"].unique().tolist()[0] == 0:
                name = 'not_cancer_{}'.format(ind)
                dict_[name] = list(set(snp_info["SNPS"].tolist()))

    g = nx.Graph()
    edges = []

    for ind, (key, val) in enumerate(dict_.items()):
        for ind1, (key1, val1) in enumerate(dict_.items()):
            if ind != ind1:
                # select diseases which have no SNPs in common
                if len(set(val).intersection(set(val1))) == 0:
                    edges.append((key, key1))

    edges = list(set(edges))
    g.add_edges_from(edges)
    color_map = []
    for node in g.nodes:
        if node.startswith('cancer'):
            color_map.append('red')
        else:
            color_map.append('violet')

    options = {'node_size': 15, 'width': 15}
    ax.set_title('{} SNPs'.format(NAMES[index - 1]), fontsize=8)

    pos = nx.fruchterman_reingold_layout(g)
    nx.draw(g, pos, node_color=color_map, **options)


keys = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'missense_variant', 'stop_gained',
        'non_coding_transcript_exon_variant', '3_prime_UTR_variant', '5_prime_UTR_variant', 'TF_binding_site_variant']

randomize = True  # False
if randomize:
    dataset_['SNPS'] = dataset_["SNPS"].sample(frac=1).values

styles = ["cancer_non_cancer", "cancer", "non_cancer"]
styles_dict = {"cancer_non_cancer": "Cancer/Non-cancer Interactions Graph", "cancer": "Cancer Interactions Graph",
               "non_cancer": "Non-cancer Interactions Graph"}
for style_ in styles:
    for i in range(1, 10):
        subax = plt.subplot(330 + i)
        draw_graph(keys[i - 1], dataset_, subax, i, 0, style_)
    # https://networkx.org/documentation/stable/tutorial.html
    red_patch = mpatches.Patch(color='red', label='cancer disease')
    violet_patch = mpatches.Patch(color='violet', label='no-cancer disease')
    plt.legend(handles=[red_patch, violet_patch], prop={'size': 7})
    fig = plt.gcf()
    if randomize:
        fig.suptitle("randomized_{}".format(styles_dict[style_]), fontsize=14)
        plt.savefig("pictures/random_{}.png".format(style_))
    else:
        fig.suptitle("{}".format(styles_dict[style_]), fontsize=14)
        plt.savefig("pictures/{}.png".format(style_))
    plt.show()