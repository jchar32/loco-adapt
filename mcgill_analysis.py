# %% Pain Character Analysis - McGill Pain Questionnaire

import re
import matplotlib.pyplot as plt

# from wordcloud import WordCloud
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

path = "../painmap mcgill data.xlsx"
df = pd.read_excel(path, header=0, sheet_name="Summary")

# %%
word_counts = {"total": {}, "t1": {}, "t2": {}, "t3": {}}
for w in df["Term"]:
    word_counts["total"][w] = df[df["Term"] == w]["Total"].values[0]
    word_counts["t1"][w] = df[df["Term"] == w]["pain1_n"].values[0]
    word_counts["t2"][w] = df[df["Term"] == w]["pain3_n"].values[0]
    word_counts["t3"][w] = df[df["Term"] == w]["pain5_n"].values[0]

# %% Bar graph of top 10 words

top10 = df.loc[:10, :].to_dict()
fig = go.Figure(
    data=[
        go.Bar(
            x=df["Term"][:10],
            y=df["pain5_n"][:10],
            textposition="auto",
            name="Pain 5/10",
            marker=dict(color=px.colors.qualitative.Dark24[0]),
        ),
        go.Bar(
            x=df["Term"][:10],
            y=df["pain3_n"][:10],
            textposition="auto",
            name="Pain 3/10",
            marker=dict(color=px.colors.qualitative.Dark24[7]),
        ),
        go.Bar(
            x=df["Term"][:10],
            y=df["pain1_n"][:10],
            textposition="auto",
            name="Pain 1/10",
            marker=dict(color=px.colors.qualitative.Dark24[10]),
        ),
    ],
)

fig.update_layout(
    barmode="stack",
    title="Top 10 Selected Words",
    yaxis_title="Frequency",
    legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
    font_family="Arial, sans-serif",
    font_size=20,
    template="simple_white",
)
fig.show()


# fig.write_html("../data/painmap/mcgill_word_freq.html")
# fig.write_image("../data/painmap/mcgill_word_freq.svg")
fig.write_image(
    "../data/painmap/mcgill_word_freq2.png",
    format="png",
    engine="kaleido",
    scale=10,
    width=1100,
    height=400,
)
# %%

# %%
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(word_counts["total"])
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.show()  # Create a bar chart of word frequencies
# plt.figure(figsize=(10, 6))
# plt.bar(list(word_counts["total"].keys()), list(word_counts["total"].values()))
# plt.xlabel("Words")
# plt.ylabel("Frequency")
# plt.title("Word Frequencies")
# plt.xticks(rotation=45)
plt.show()
