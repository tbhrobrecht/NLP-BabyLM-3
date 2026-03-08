import re
import matplotlib.pyplot as plt

# Paste your data as a multiline string
data = """
BA_BEI_subj_drop                                    21.67%        65/300 
BA_deletion                                         95.00%       285/300 
BA_duplicate_argument                                0.00%         0/300 
BA_inversion                                        36.67%       110/300 
BA_meiba                                             0.00%         0/300 
BA_negation                                         55.33%       166/300 
BA_no_progressive                                    2.00%         6/300 
BA_no_stative_verb                                  99.33%       298/300 
BA_suo_adverbial_a                                  11.67%        35/300 
BA_suo_adverbial_b                                  13.00%        39/300 
BA_verb_le_a                                         0.00%         0/300 
BA_verb_le_b                                       100.00%       300/300 
BEI_construction_a                                  43.33%       130/300 
BEI_construction_b                                  42.67%       128/300 
BEI_deletion                                        98.00%       294/300 
BEI_preposition                                     30.00%        90/300 
PN_numP_a                                           15.00%        45/300 
PN_numP_b                                           72.67%       218/300 
adjective_transitive_dui                            87.00%       261/300 
agent_animacy_adv                                   70.67%       212/300 
agent_animacy_passive                               69.33%       208/300 
agent_animacy_subj                                  51.33%       154/300 
agent_causative                                     54.33%       163/300 
agent_deletion                                      80.67%       242/300 
anaphor_gender_agreement                            52.33%       157/300 
anaphor_number_agreement                             0.00%         0/300 
causative_shi_ba                                     2.67%         8/300 
classifier_noun_agreement                           58.33%       175/300 
classifier_noun_agreement_no_gap                    46.33%       139/300 
control_modal_vs_raising_modal                      59.33%       178/300 
ellipsis_adj                                        54.33%       163/300 
ellipsis_double_object                               0.00%         0/300 
ellipsis_n_bar_class                                44.33%       133/300 
existential_there_subject_raising                   18.00%        54/300 
fci_renhe_dou                                      100.00%       300/300 
fci_renhe_prepP                                      0.67%         2/300 
fci_renhe_ruguo                                      0.00%         0/300 
fci_renhe_subj                                       0.00%         0/300 
fci_renhe_suoyou                                     0.00%         0/300 
intransitive_double_obj                              0.00%         0/300 
intransitive_no_obj                                 47.67%       143/300 
left_adverbial_b                                    42.67%       128/300 
left_adverbial_d                                    45.00%       135/300 
left_adverbial_e                                    26.33%        79/300 
left_adverbial_negation                             71.67%       215/300 
left_dou                                            80.67%       242/300 
modal_raising_hui                                   44.67%       134/300 
modal_raising_topicalization                        63.33%       190/300 
nominal_definite_men                                 7.00%        21/300 
nominal_modal_insertion                             45.33%       136/300 
noun_adjective_shi                                 100.00%       300/300 
npi_renhe_A_not_A_question                           0.00%         0/300 
npi_renhe_conditional                              100.00%       300/300 
npi_renhe_neg_scope_locP                            64.00%       192/300 
npi_renhe_neg_scope_subj                            93.00%       279/300 
npi_renhe_wh_question_obj                          100.00%       300/300 
npi_renhe_wh_question_subj                          22.00%        66/300 
passive_agent_deletion_long_left                    64.33%       193/300 
passive_agent_deletion_long_right_b                 72.33%       217/300 
passive_agent_deletion_short                        76.33%       229/300 
passive_body_part                                   65.00%       195/300 
passive_intransitive                                58.00%       174/300 
passive_no_adj                                      54.67%       164/300 
passive_suo                                         76.00%       228/300 
plural_cardinal_men_a                                3.67%        11/300 
plural_cardinal_men_b                                5.33%        16/300 
preposition_deletion                                97.67%       293/300 
preposition_insertion                               95.67%       287/300 
principle_A_c_command                                0.00%         0/300 
principle_A_c_command_number                         0.00%         0/300 
principle_A_domain                                   0.00%         0/300 
principle_A_domain_number                          100.00%       300/300 
question_A_not_A                                    11.33%        34/300 
question_A_not_A_daodi_a                            45.67%       137/300 
question_A_not_A_daodi_b                            41.33%       124/300 
question_A_not_A_indirect                            0.00%         0/300 
question_V_not_VP_1                                  0.00%         0/300 
question_V_not_VP_2                                 24.33%        73/300 
question_daodi_nandao_1                             95.00%       285/300 
question_daodi_nandao_2                              6.33%        19/300 
question_daodi_nandao_A_not_A_intran                 5.33%        16/300 
question_daodi_nandao_A_not_A_tran                   7.33%        22/300 
question_daodi_negation                             16.33%        49/300 
question_nandao_negation                            40.00%       120/300 
question_nandao_raising_1_a                         59.67%       179/300 
question_nandao_raising_1_b                         81.33%       244/300 
question_nandao_raising_2                           89.33%       268/300 
question_nandao_raising_3                           83.00%       249/300 
question_nandao_scope_1                             77.67%       233/300 
question_nandao_scope_2                             70.33%       211/300 
question_particle_daodi_choice_intran               72.33%       217/300 
question_particle_daodi_choice_tran                 50.67%       152/300 
question_particle_nandao                            88.00%       264/300 
relative_operator_intepretation                    100.00%       300/300 
relative_operator_who                               93.33%       280/300 
relativization_movement_no_gap                       0.00%         0/300 
relativization_movement_when_where                   5.33%        16/300 
renhe_no_episodic_sentences                         17.67%        53/300 
renhe_no_superordinate_negation                     99.33%       298/300 
renhe_non_factive_verb                             100.00%       300/300 
right_yijing_a                                      90.00%       270/300 
right_yijing_b                                      66.00%       198/300 
singular_PN_but_plural_pron                          7.00%        21/300 
superlative_quantifiers_1                           37.67%       113/300 
superlative_quantifiers_2                          100.00%       300/300 
topicalization_OSV                                  58.33%       175/300 
topicalization_OSV_mei                              55.33%       166/300 
topicalization_SOV_mei                              62.67%       188/300 
verb_negation_particle                              44.33%       133/300 
verb_phrase_left_adverbial                          45.67%       137/300 
verb_phrase_left_negation                           63.33%       190/300 
ya_insertion                                        52.33%       157/300 
you_quantifier_adj                                  50.00%       150/300 
you_yige                                             0.00%         0/300 
"""

# Parse the data
labels = []
accuracies = []

for line in data.strip().split('\n'):
    match = re.match(r"([A-Za-z0-9_]+)\s+([0-9.]+)%", line)
    if match:
        labels.append(match.group(1))
        accuracies.append(float(match.group(2)))

# Filter out 50% accuracy entries
filtered = [(label, acc) for label, acc in zip(labels, accuracies) if acc > 50.0]
labels, accuracies = zip(*filtered)


# Improved blue color scheme: 100% = darkest, 80%+ = regular, <80% = light
colors = []
for acc in accuracies:
    if acc == 100.0:
        colors.append('#08306b')  # very dark blue
    elif acc >= 80.0:
        colors.append('#2171b3')  # regular blue
    else:
        colors.append('#a6bddb')  # light blue



# Bar graph (first plot)
# plt.figure(figsize=(18, 6))
# plt.bar(labels, accuracies, color=colors)
# plt.ylabel('Accuracy (%)')
# plt.xlabel('Subset')
# plt.title('BLiMP Subset Accuracies (Excluding <50%)')
# plt.xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=8)
# plt.xlim(-0.5, len(labels) - 0.5)
# plt.tight_layout()
# plt.show()


# Box and whiskers plot (second plot) -- use all accuracy values
all_accuracies = [float(match.group(2)) for line in data.strip().split('\n') if (match := re.match(r"([A-Za-z0-9_]+)\s+([0-9.]+)%", line))]

# plt.figure(figsize=(10, 2.5))
# plt.boxplot(all_accuracies, vert=False, patch_artist=True,
#             boxprops=dict(facecolor='#a6bddb', color='#2171b3'),
#             medianprops=dict(color='#08306b', linewidth=2),
#             whiskerprops=dict(color='#2171b3'),
#             capprops=dict(color='#2171b3'),
#             flierprops=dict(markerfacecolor='red', marker='o', markersize=6, linestyle='none'))
# plt.xlabel('Accuracy (%)')
# plt.yticks([])
# plt.title('Distribution of Accuracies (Box & Whiskers)')
# plt.tight_layout()
# plt.show()


# Histogram plot (all accuracies) with more details
import numpy as np
from scipy.stats import gaussian_kde
mean_acc = np.mean(all_accuracies)
median_acc = np.median(all_accuracies)
min_acc = np.min(all_accuracies)
max_acc = np.max(all_accuracies)


plt.figure(figsize=(10, 4))
n, bins, patches = plt.hist(all_accuracies, bins=20, color='#2171b3', edgecolor='black', alpha=0.7)
plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_acc:.2f}%')
plt.axvline(median_acc, color='darkgreen', linestyle='dotted', linewidth=1.5, label=f'Median: {median_acc:.2f}%')
percentile_25 = np.percentile(all_accuracies, 25)
percentile_75 = np.percentile(all_accuracies, 75)
xs = np.linspace(min_acc, max_acc, 200)
# KDE (line of best fit)
# density = gaussian_kde(all_accuracies)
# scale = len(all_accuracies) * (bins[1] - bins[0])
# plt.plot(xs, density(xs) * scale, color='black', linewidth=2, label='KDE (Best Fit)')
# Parabola (2nd-degree polynomial) fit to histogram
bin_centers = 0.5 * (bins[:-1] + bins[1:])
hist_counts, _ = np.histogram(all_accuracies, bins=bins)
poly_coeffs = np.polyfit(bin_centers, hist_counts, 2)
poly_fit = np.polyval(poly_coeffs, xs)
plt.plot(xs, poly_fit, color='black', linewidth=2, label='Parabola Fit (2nd degree)')
plt.xlabel('Accuracy (%)')
plt.ylabel('Count')
plt.title('BLiMP Histogram of Accuracies')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# # Strip plot (dot plot, all accuracies) with more details
# plt.figure(figsize=(10, 2.5))
# plt.scatter(all_accuracies, [1]*len(all_accuracies), color='#2171b3', alpha=0.7, s=40, label='Accuracies')
# plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_acc:.2f}%')
# plt.axvline(median_acc, color='orange', linestyle='dotted', linewidth=1.5, label=f'Median: {median_acc:.2f}%')
# plt.xlabel('Accuracy (%)')
# plt.yticks([])
# plt.title('Strip Plot (Dot Plot) of Accuracies')
# plt.legend(loc='upper left')
# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# # Violin plot (all accuracies) with more details
# plt.figure(figsize=(10, 4))
# parts = plt.violinplot(all_accuracies, vert=False, showmeans=True, showmedians=True)
# plt.xlabel('Accuracy (%)')
# plt.yticks([])
# plt.title('Violin Plot of Accuracies')
# plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_acc:.2f}%')
# plt.axvline(median_acc, color='orange', linestyle='dotted', linewidth=1.5, label=f'Median: {median_acc:.2f}%')
# plt.legend(loc='upper left')
# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()