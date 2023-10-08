import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi


def plot_summarization_results():
    xs = range(6)
    models = ["BertSum","PEGASUS","MGSum","GraphSum","Hi-Map","HT"]

    psv1_rougel = [17.21, 32.73, 28.01, 27.02, 20.34, 28.41]
    psv2_rougel = [10.33, 33.77, 28.93, 28.78, 20.23, 27.44]
    psv3_rougel = [17.99, 31.84, 28.78, 28.01, 20.56, 28.02]
    wiki_rougel = [35.98, 37.10, 33.21, 36.97, 34.97, 35.76]
    mn_rougel = [23.01, 24.26, 20.09, 22.50, 17.41, 18.80]
    wcep_rougel = [25.99, 28.01, 26.88, 26.85, 26.14, 27.32]
    mx_rougel = [29.57, 34.57, 31.25, 33.48, 28.43, 29.96]


    psv1_bert_r = [23.10, 34.43, 30.10, 30.01, 24.32, 31.33]
    psv1_bert_p = [24.01, 33.97, 30.11, 29.56, 24.10, 31.13]
    psv2_bert_p = [22.83, 34.40, 30.01, 28.45, 24.31, 31.04]
    psv2_bert_r = [22.35, 33.96, 28.03, 28.42, 24.04, 31.03]
    psv3_bert_p = [23.19, 34.53, 30.18, 30.00, 24.42, 31.83]
    psv3_bert_r = [23.01, 34.21, 30.14, 28.94, 24.16, 31.84]
    wiki_bert_p = [28.14, 39.94, 35.43, 32.02, 26.42, 33.80]
    wiki_bert_r = [29.33, 39.98, 36.23, 31.92, 26.41, 34.59]
    mn_bert_p = [29.93, 40.86, 36.63, 32.82, 26.37, 34.44]
    mn_bert_r = [30.43, 41.67, 36.88, 32.96, 26.36, 34.77]
    wcep_bert_p = [25.49, 35.63, 32.48, 31.01, 26.43, 33.81]
    wcep_bert_r = [26.89, 36.21, 32.78, 31.73, 27.01, 33.88]
    mx_bert_p = [25.83, 35.61, 32.99, 31.51, 26.78, 34.21]
    mx_bert_r = [25.82, 34.61, 33.79, 32.64, 26.99, 35.19]

    psv1_bert_f = [(2*x*y)/(x+y) for x,y in zip(psv1_bert_p, psv1_bert_r)]
    psv2_bert_f = [(2*x*y)/(x+y) for x,y in zip(psv2_bert_p, psv2_bert_r)]
    psv3_bert_f = [(2*x*y)/(x+y) for x,y in zip(psv3_bert_p, psv3_bert_r)]
    wiki_bert_f = [(2*x*y)/(x+y) for x,y in zip(wiki_bert_p, wiki_bert_r)]
    mn_bert_f = [(2*x*y)/(x+y) for x,y in zip(mn_bert_p, mn_bert_r)]
    wcep_bert_f = [(2*x*y)/(x+y) for x,y in zip(wcep_bert_p, wcep_bert_r)]
    mx_bert_f = [(2*x*y)/(x+y) for x,y in zip(mx_bert_p, mx_bert_r)]

    plt.figure(figsize=(9,3.5))
    plt.subplot(1, 2, 1)
    line1, = plt.plot(xs, psv1_rougel, color='deepskyblue', linestyle = '-', linewidth='1.5')
    line2, = plt.plot(xs, psv2_rougel, color="deepskyblue", linestyle = ':', linewidth='1.5')
    line3, = plt.plot(xs, psv3_rougel, color="deepskyblue", linestyle = '--', linewidth='1.5')
    line4, = plt.plot(xs, wiki_rougel, color="darkcyan", linestyle="-", linewidth='1.5')
    line5, = plt.plot(xs, mn_rougel, color="orange", linestyle='-', linewidth='1.5')
    line6, = plt.plot(xs, wcep_rougel, color="brown", linestyle='--', linewidth='1.5')
    line7, = plt.plot(xs, mx_rougel, color="magenta", linestyle="--", linewidth='1.5')
    plt.xticks(xs, models, rotation=25, family='Times New Roman', fontsize=16)
    plt.ylabel(r'$\mathrm{Rouge-L}$', fontsize=16, family='Times New Roman')
    plt.subplot(1, 2, 2)
    line1, = plt.plot(xs, psv1_bert_f, color='deepskyblue', linestyle = '-', linewidth='1.5', label="PeerSum-R")
    line2, = plt.plot(xs, psv2_bert_f, color="deepskyblue", linestyle = ':', linewidth='1.5', label="PeerSum-RC")
    line3, = plt.plot(xs, psv3_bert_f, color="deepskyblue", linestyle = '--', linewidth='1.5', label="PeerSum-ALL")
    line4, = plt.plot(xs, wiki_bert_f, color="darkcyan", linestyle="-", linewidth='1.5', label="Wikisum")
    line5, = plt.plot(xs, mn_bert_f, color="orange", linestyle='-', linewidth='1.5', label="Multi-News")
    line6, = plt.plot(xs, wcep_bert_f, color="brown", linestyle='--', linewidth='1.5', label="WCEP")
    line7, = plt.plot(xs, mx_bert_f, color="magenta", linestyle="--", linewidth='1.5', label="Multi-XScience")
    plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7], bbox_to_anchor=(-0.17, -0.47), loc=10, borderaxespad=0,
               ncol=4, prop={"family":'Times New Roman', "size":13.5})
    plt.xticks(xs, models, rotation=25, family='Times New Roman', fontsize=16)
    plt.ylabel("$\mathrm{F}_\mathrm{BERT}$", fontsize=16, family='Times New Roman')
    plt.subplots_adjust(top=0.97, bottom=0.38)
    plt.show()
    # plt.savefig('summarization_results.png', dpi=1024)


def plot_summarization_results_bar_chart():
    models = ["BertSum","PEGASUS","MGSum","GraphSum","Hi-Map","HT"]
    xs = range(1, len(models)+1)
    xs_psv1 = (np.array(xs) - 0.3).tolist()
    xs_psv2 = (np.array(xs) - 0.2).tolist()
    xs_psv3 = (np.array(xs) - 0.1).tolist()
    xs_wiki = (np.array(xs)).tolist()
    xs_mn = (np.array(xs) + 0.1).tolist()
    xs_wcep = (np.array(xs) + 0.2).tolist()
    xs_mx = (np.array(xs) + 0.3).tolist()
    print(xs_psv1)

    psv1_rougel = [17.21, 32.73, 28.01, 27.02, 20.34, 28.41]
    psv2_rougel = [10.33, 33.77, 28.93, 28.78, 20.23, 27.44]
    psv3_rougel = [17.99, 31.84, 28.78, 28.01, 20.56, 28.02]
    wiki_rougel = [35.98, 37.10, 33.21, 36.97, 34.97, 35.76]
    mn_rougel = [23.01, 24.26, 20.09, 22.50, 17.41, 18.80]
    wcep_rougel = [25.99, 28.01, 26.88, 26.85, 26.14, 27.32]
    mx_rougel = [29.57, 34.57, 31.25, 33.48, 28.43, 29.96]

    psv1_bert_r = [23.10, 34.43, 30.10, 30.01, 24.32, 31.33]
    psv1_bert_p = [24.01, 33.97, 30.11, 29.56, 24.10, 31.13]
    psv2_bert_p = [22.83, 34.40, 30.01, 28.45, 24.31, 31.04]
    psv2_bert_r = [22.35, 33.96, 28.03, 28.42, 24.04, 31.03]
    psv3_bert_p = [23.19, 34.53, 30.18, 30.00, 24.42, 31.83]
    psv3_bert_r = [23.01, 34.21, 30.14, 28.94, 24.16, 31.84]
    wiki_bert_p = [28.14, 39.94, 35.43, 32.02, 26.42, 33.80]
    wiki_bert_r = [29.33, 39.98, 36.23, 31.92, 26.41, 34.59]
    mn_bert_p = [29.93, 40.86, 36.63, 32.82, 26.37, 34.44]
    mn_bert_r = [30.43, 41.67, 36.88, 32.96, 26.36, 34.77]
    wcep_bert_p = [25.49, 35.63, 32.48, 31.01, 26.43, 33.81]
    wcep_bert_r = [26.89, 36.21, 32.78, 31.73, 27.01, 33.88]
    mx_bert_p = [25.83, 35.61, 32.99, 31.51, 26.78, 34.21]
    mx_bert_r = [25.82, 34.61, 33.79, 32.64, 26.99, 35.19]

    psv1_bert_f = [(2*x*y)/(x+y) for x,y in zip(psv1_bert_p, psv1_bert_r)]
    psv2_bert_f = [(2*x*y)/(x+y) for x,y in zip(psv2_bert_p, psv2_bert_r)]
    psv3_bert_f = [(2*x*y)/(x+y) for x,y in zip(psv3_bert_p, psv3_bert_r)]
    wiki_bert_f = [(2*x*y)/(x+y) for x,y in zip(wiki_bert_p, wiki_bert_r)]
    mn_bert_f = [(2*x*y)/(x+y) for x,y in zip(mn_bert_p, mn_bert_r)]
    wcep_bert_f = [(2*x*y)/(x+y) for x,y in zip(wcep_bert_p, wcep_bert_r)]
    mx_bert_f = [(2*x*y)/(x+y) for x,y in zip(mx_bert_p, mx_bert_r)]

    bar_width = 0.1

    plt.figure(figsize=(9,3.5))
    plt.subplot(1, 2, 1)
    plt.bar(xs_psv1, height=psv1_rougel, width=bar_width, color='deepskyblue')
    plt.bar(xs_psv2, height=psv2_rougel, width=bar_width, color='darkcyan')
    plt.bar(xs_psv3, height=psv3_rougel, width=bar_width, color="orange")
    plt.bar(xs_wiki, height=wiki_rougel, width=bar_width, color='brown')
    plt.bar(xs_mn, height=mn_rougel, width=bar_width, color='magenta')
    plt.bar(xs_wcep, height=wcep_rougel, width=bar_width, color='navy')
    plt.bar(xs_mx, height=mx_rougel, width=bar_width, color='darksalmon')
    plt.xticks(xs, models, rotation=25, family='Times New Roman', fontsize=16)
    plt.ylabel(r'$\mathrm{Rouge-L}$', fontsize=16, family='Times New Roman')
    plt.subplot(1, 2, 2)
    plt.bar(xs_psv1, height=psv1_bert_f, width=bar_width, color='deepskyblue', label="PeerSum-R")
    plt.bar(xs_psv2, height=psv2_bert_f, width=bar_width, color='darkcyan', label="PeerSum-RC")
    plt.bar(xs_psv3, height=psv3_bert_f, width=bar_width, color='orange', label="PeerSum-ALL")
    plt.bar(xs_wiki, height=wiki_bert_f, width=bar_width, color='brown', label="Wikisum")
    plt.bar(xs_mn, height=mn_bert_f, width=bar_width, color='magenta', label="Multi-News")
    plt.bar(xs_wcep, height=wcep_bert_f, width=bar_width, color='navy', label="WCEP")
    plt.bar(xs_mx, height=mx_bert_f, width=bar_width, color='darksalmon', label="Multi-XScience")
    plt.legend(bbox_to_anchor=(-0.17, -0.47), loc=10, borderaxespad=0, ncol=4, prop={"family":'Times New Roman', "size":13.5})
    plt.xticks(xs, models, rotation=25, family='Times New Roman', fontsize=16)
    plt.ylabel("$\mathrm{F}_\mathrm{BERT}$", fontsize=16, family='Times New Roman')
    plt.subplots_adjust(top=0.97, bottom=0.38)
    # plt.show()
    plt.savefig('summarization_results.png', dpi=1024)


#, 'Multi-News', 'WCEP', 'Multi-XScience'
def plot_radar_chart():
    categories = ['Unigram', 'Bigram', 'Trigram', 'Relevance']
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(8, 3.5))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, fontsize=16, family='Times New Roman')

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([40, 60, 80], ["40", "60", "80"], color="grey", size=16)
    plt.ylim(0, 100)

    ax.plot(angles, [35.34, 80.29, 90.92, 42.9, 35.34], linewidth=1.5, linestyle='solid', label="PeerSum-R")
    ax.plot(angles, [], linewidth=1.5, linestyle='solid', label="PeerSum-RC")
    ax.plot(angles, [], linewidth=1.5, linestyle='solid', label="PeerSum-ALL")
    ax.plot(angles, [], linewidth=1.5, linestyle='solid', label="WikiSum")
    ax.plot(angles, [], linewidth=1.5, linestyle='solid', label="Multi-News")
    ax.plot(angles, [], linewidth=1.5, linestyle='solid', label="WCEP")
    ax.plot(angles, [], linewidth=1.5, linestyle='solid', label="Multi-XScience")
    # ax.fill(angles, values, 'b', alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(-0.25, 0.6), prop={"family":'Times New Roman', "size":12})
    # plt.show()
    plt.savefig('radar_chart.png', dpi=1024)


def plot_score_based_evaluation():
    models = ['Ground truth', 'MTSum', 'MTSum-D', 'MTSum-E', 'BART', 'LED', 'PEGASUS', 'GraphSum', 'PRIMERA']
    bar_colors = ['deepskyblue', 'darkcyan', 'magenta', 'darksalmon', 'brown', 'darkgrey', 'orange', 'limegreen',
                  'cornflowerblue']
    ys = [64.92, 62.70, 1, 1, 1, 1, 1, 1, 1]
    xs = range(1, len(models) + 1)
    bar_width = 0.6
    plt.figure(figsize=(9, 4))
    plt.subplot()
    plt.bar(xs, height=ys, width=bar_width, color=bar_colors)
    plt.xticks(xs, models, rotation=30, family='Times New Roman', fontsize=22)
    plt.yticks(fontproperties='Times New Roman', fontsize=22)
    plt.ylabel('Mean-Square Error', fontdict={"family":'Times New Roman', "size":24})
    plt.subplots_adjust(top=0.97, bottom=0.27)
    # plt.show()
    plt.savefig('score_evaluation.png', dpi=1024)


if __name__=="__main__":
    plot_score_based_evaluation()