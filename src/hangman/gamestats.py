# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hangman.csv", header=None, 
                 names=["WORD_LEN", "SOLVER_WINS", "NUM_GUESSES"])

# wins vs losses
nloss = df[df["SOLVER_WINS"] == 0].count()["SOLVER_WINS"]
nwins = df[df["SOLVER_WINS"] == 1].count()["SOLVER_WINS"]
print "probability of winning=", nwins / (nwins + nloss)
print "probability of losing=", nloss / (nwins + nloss)

## probability of winning and losing for different word lengths
df2 = df.drop("NUM_GUESSES", 1)
df2_wins = df2[df2["SOLVER_WINS"] == 1].groupby("WORD_LEN").count().reset_index()
df2_losses = df2[df2["SOLVER_WINS"] == 0].groupby("WORD_LEN").count().reset_index()
df2_losses.rename(columns={"SOLVER_WINS": "SOLVER_LOSES"}, inplace=True)
df2_merged = df2_wins.merge(df2_losses, how="inner", on="WORD_LEN")
df2_merged.plot(kind="bar", stacked=True, x="WORD_LEN", 
                title="Win/Loss Counts by Word Length") 
plt.show()
df2_merged["NUM_GAMES"] = df2_merged["SOLVER_WINS"] + df2_merged["SOLVER_LOSES"]
df2_merged["SOLVER_WINS"] = df2_merged["SOLVER_WINS"] / df2_merged["NUM_GAMES"]
df2_merged["SOLVER_LOSES"] = df2_merged["SOLVER_LOSES"] / df2_merged["NUM_GAMES"]
df2_merged.drop("NUM_GAMES", axis=1, inplace=True)
df2_merged.plot(kind="bar", stacked=True, x="WORD_LEN", 
                title="Win/Loss Probabilities by Word Length") 
plt.show()

# how number of guesses to win varies with word length (winning games only)
df3 = df[df["SOLVER_WINS"] == 1]
df3.drop("SOLVER_WINS", 1)
df3_grouped = df3.drop("SOLVER_WINS", 1).groupby("WORD_LEN").mean().reset_index()
df3_grouped.plot(kind="bar", x="WORD_LEN", 
                 title="Avg Guesses for Different Word Lengths")
plt.show()
