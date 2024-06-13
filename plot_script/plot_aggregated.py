import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
df_front = pd.read_csv('all_front.csv')
df_longhorn = pd.read_csv('all_longhorn.csv')

# Graphing customizability options
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('seaborn-whitegrid')
sns.set_palette("colorblind")

# Setup the matplotlib figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(18, 4), sharey=True)

# Font size settings
label_font_size = 18  # For x and y labels
tick_font_size = 14   # For x and y ticks
legend_font_size = 14  # For legend

# Custom labels for x-axis
custom_labels = ['ResNet50', 'BERT', 'PageRank']

# First plot (Frontera)
sns.boxplot(x='exp', y='mean_iter_dur_median_normalized', showfliers=False,
            data=df_front, palette="Set2", notch=False, medianprops={'color': 'black'}, ax=axes[0])
sns.stripplot(x='exp', y='mean_iter_dur_median_normalized',
              hue='cabinet', data=df_front, size=5, jitter=True, dodge=False, ax=axes[0])
# axes[0].set_title(
#     'Normalized Performance Variability per Model in Frontera')
axes[0].set_ylabel('Normalized Performance', fontsize=label_font_size)
axes[0].set_xlabel('Models', fontsize=label_font_size)
axes[0].set_xticklabels(custom_labels, fontsize=tick_font_size)
axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=tick_font_size)
axes[0].grid(True)


# Second plot (Longhorn)
sns.boxplot(x='exp', y='mean_iter_dur_median_normalized', showfliers=False,
            data=df_longhorn, palette="Set2", notch=False, medianprops={'color': 'black'}, ax=axes[1])
sns.stripplot(x='exp', y='mean_iter_dur_median_normalized',
              hue='cabinet', data=df_longhorn, size=5, jitter=True, dodge=False, ax=axes[1])

axes[1].set_xlabel('Models')
axes[1].set_ylabel('Normalized Performance', fontsize=label_font_size)
axes[1].set_xlabel('Models', fontsize=label_font_size)
axes[1].set_xticklabels(custom_labels, fontsize=tick_font_size)
axes[1].set_yticklabels(axes[0].get_yticks(), fontsize=tick_font_size)
axes[1].grid(True)

# Adjust legend
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=legend_font_size)

# Adjust the layout
plt.tight_layout()

# Save and show the plot
plt.savefig('perf_front_longhorn.pdf', bbox_inches='tight')
plt.close(fig)

# Graphing customizability options
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('seaborn-whitegrid')
sns.set_palette("colorblind")

fig_ft, ax = plt.subplots(1, 1, figsize=(8, 4), sharey=True)
# First plot (Frontera)
sns.boxplot(x='exp', y='mean_iter_dur_median_normalized', showfliers=False,
            data=df_front, palette="Set2", notch=False, medianprops={'color': 'black'}, ax=ax)
sns.stripplot(x='exp', y='mean_iter_dur_median_normalized',
              hue='cabinet', data=df_front, size=5, jitter=True, dodge=False, ax=ax)

ax.set_ylabel('Normalized Performance', fontsize=label_font_size)
ax.set_xlabel('Models', fontsize=label_font_size)
ax.set_xticklabels(custom_labels, fontsize=tick_font_size)
ax.set_yticklabels(axes[0].get_yticks(), fontsize=tick_font_size)
ax.grid(True)
plt.savefig('perf_frontera.pdf', bbox_inches='tight')
plt.close(fig)

# Graphing customizability options
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('seaborn-whitegrid')
sns.set_palette("colorblind")

fig_ft, ax = plt.subplots(1, 1, figsize=(8, 4), sharey=True)
# First plot (Frontera)
sns.boxplot(x='exp', y='mean_iter_dur_median_normalized', showfliers=False,
            data=df_longhorn, palette="Set2", notch=False, medianprops={'color': 'black'}, ax=ax)
sns.stripplot(x='exp', y='mean_iter_dur_median_normalized',
              hue='cabinet', data=df_longhorn, size=5, jitter=True, dodge=False, ax=ax)

ax.set_ylabel('Normalized Performance', fontsize=label_font_size)
ax.set_xlabel('Models', fontsize=label_font_size)
ax.set_xticklabels(custom_labels, fontsize=tick_font_size)
ax.set_yticklabels(axes[0].get_yticks(), fontsize=tick_font_size)
ax.grid(True)
plt.savefig('perf_longhorn.pdf', bbox_inches='tight')
plt.close(fig)

# metric = 'mean_iter_dur_median_normalized'
# q1 = df[metric].quantile(0.25)
# q2 = df[metric].quantile(0.50)
# q3 = df[metric].quantile(0.75)
# iqr = q3 - q1
# range = q3 + 1.5 * iqr - (q1 - 1.5 * iqr)
# percent_variability = range/q2 * 100

# print('Q1: ' + str(q1) + ' Q2: ' + str(q2) + ' Q3: ' + str(q3) +
#       ' Percent Variability: ' + str(percent_variability) + '%')

# filtered_df = df[((df['mean_iter_dur_median_normalized'] > (
#     q1 - 1.5 * iqr)) | (df['mean_iter_dur_median_normalized'] < (q3 + 1.5 * iqr)))]
# ordered_df = filtered_df.sort_values(
#     by='mean_iter_dur_median_normalized', ascending=False)[['exp', 'cabinet', 'node', 'device', 'mean_iter_dur_median_normalized']]
# print(ordered_df.head(10))

