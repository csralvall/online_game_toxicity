# Toxicity tipification in online game chats
---
 > Project developed for the course Text Mining, given by Dr. Laura Alonso Allemany, FAMAF-UNC, 2021.
 
 > Author: César Alvarez Vallero
### Introduction
---
In the last 20 years the grow of online gaming has increased the human communication through online
chats inside games. The mix of high competitiveness and relative anonimity have made a perfect
place for the development of toxicity.

The goal of this work is to detect types of toxicity inside chat conversations from online games.
Some inspiration was taken from the research initiative founded by Jigsaw and Google (Alphabet
subsidiaries) and their kaggle competitions, were they incentivize the creation of models to detect
toxicity in the comments of wikipedia edtion section. The initial classification provided by the
challenges settled a baseline to classify types of toxicity:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Although the current work tries to avoid using this as a framework rather than a guideline.

### Dataset
---
The dataset used is publicly available in [kaggle](https://www.kaggle.com/romovpa/gosuai-dota-2-game-chats).
It contains chats of almost 1 million matches from public matchmaking of the game Dota2 by Valve.

To perform the task a subset of the whole dataset was selected. That subset contained all the english
messages inside the dataset. Its total size is around 60 million messages.

Dota2 is an online game that belongs to the genre Multiplayer Online Battle Arena (MOBA). It involves
two teams and an objective that implies the defeat of the opposite team.

The initial structure of the dataset was the following:

|match|time|slot|text|
|-----|----|----|----|

Where:
- _match_ is the index
- _time_ is the game time when the message was sent
- _slot_ is the player postion in the arena (0–4 for Radiant, 5–9 for Dire)
- _text_ is the message

Some related works using this dataset are:
- [Measuring toxicity in Dota 2](https://www.h4pz.co/dota-2-toxicity)
- [Toxicity detection in gaming](https://databricks.com/notebooks/toxic-test-gam/index.html#01_intro.html)


### Preprocessing
---
As the dataset contained almost a million matches with at least one hundred messages each one, the size of data
became unmanagable rapidly and required different approachs to overcome that difficulty. Most of the techniques
used involved parallelization.

The task performed over the data can be seen in this [notebook](https://github.com/csralvall/online_game_toxicity/blob/main/preprocessing.ipynb).

The main points to highlight of this section are:
- Language detection
- Message cleaning: Remove of special characters with regex
- Message filtering: Use of threshold and stop words to select messages
- Feature extraction:
  - Intensity: presence of capital letters and exclamation signs
  - Toxicity: presence of "bad words" obtained from a [web list](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt)


### Approaches
---
#### Bag of words clustering
---
#### Word2Vec clustering
---
#### FastText clustering
---
### Results
---
### Conclusion
