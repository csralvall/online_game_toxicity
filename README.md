# Toxicity tipification in online game chats
---
 > Subject: Text Mining, FAMAF-UNC, 2021.

 > Teacher: Laura Alonso Allemany

 > Author: César Alvarez Vallero


<details>
  <summary>Introduction</summary>

### Introduction
--- 
In the last 20 years the grow of online gaming has increased the human
communication through online chats inside games. The mix of high
competitiveness and relative anonimity have made a perfect place for the
development of toxicity.

The goal of this work is to detect types of toxicity inside chat conversations
from online games. Some inspiration was taken from the research initiative
founded by Jigsaw and Google (Alphabet subsidiaries) and their kaggle
competitions, were they incentivize the creation of models to detect toxicity
in the comments of wikipedia edtion section. The initial classification
provided by the challenges settled a baseline to classify types of toxicity:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Although the current work tries to avoid using this as a framework rather than
a guideline.

</details>

<details>
  <summary>Dataset</summary>

### Dataset
--- 
The dataset used is publicly available in
[kaggle](https://www.kaggle.com/romovpa/gosuai-dota-2-game-chats). It contains
chats of almost 1 million matches from public matchmaking of the game Dota2 by
Valve.

To perform the task a subset of the whole dataset was selected. That subset
contained all the english messages inside the dataset. Its total size is around
60 million messages.

Dota2 is an online game that belongs to the genre Multiplayer Online Battle
Arena (MOBA). It involves two teams and an objective that implies the defeat of
the opposite team.

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

</details>

<details>
  <summary>Preprocessing</summary>

### Preprocessing
---
As the dataset contained almost a million matches with at least one hundred
messages each one, the size of data became unmanagable rapidly and required
different approachs to overcome that difficulty. Most of the techniques used
involved parallelization.

The task performed over the data can be seen in this
[notebook](./preprocessing.ipynb).

The main points to highlight of this section are:
- Language detection
- Message cleaning: Remove of special characters with regex
- Message filtering: Use of threshold and stop words to select messages
- Feature extraction:
  - Intensity: presence of capital letters and exclamation signs
  - Toxicity: presence of "bad words" obtained from a [web list](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt)

</details>

<details>
  <summary>Approaches</summary>

### Approaches
---
The approach for this task was datadriven and it relies heavily in clustering,
with some variations in the generation of the message representation.

The number of english chats in the dataset was around 60 million, so, again, a
subset of 10000 was taken from it, because a bigger sample required more time
that the disposed to achieve results.

The unit selected to clusterize were the messages. Each unit was vectorized
with one of the following techniques:
- Bag of words.
- Word2Vec embeddings.
- FastText embeddings.

Those vectors were concatenated with a vector that represented the number of
ocurrences in a list of [bad words](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt)
and the intensity score for that message.

All vectors were normalized and features with high correlation were removed.

<details>
  <summary>Bag of Words</summary>

#### Bag of words clustering
--- 
To explore further go to the [notebook](./clustering_bow.ipynb).

The number of clusters choosen for this approach was 50. It was enough to test
the capabilities of the mehtod.

The distribution of chats in clusters was the following:

![Cluster distribution - Bag of Words](./images/clusters_bow.png)

As you can see, for this clustering approach and this number of clusters, there
is one cluster that reunites all those messages that are unique or couldn't be
differentiated clearly of the rest of them. That cluster and all the messages
in it were discarded for the current analyisis.

Each message was annotated with their cluster, and then the reduced dataset
was grouped by cluster:

```python
bow_group = df_test.groupby('bow_clusters')
```

For each cluster a toxicity score was computed:

```python
bow_score = (bow_group['toxicity'].sum() / bow_group['nwords'].sum())
bow_scored = pd.DataFrame({'score': bow_score.values, 'size': bow_group.size()})
```
Then, they were sorted and filtered to get the cluster with the greatest score:

```python
bow_scored = bow_scored[bow_scored['score'] > 0.14]
bow_scored = bow_scored.sort_values(by=['score', 'size'], ascending=[False, False])
```

<a name="bow-score-table" />
The following results were obtained from the previous code:

|bow_clusters|    score   |    size    |
|------------|-----------:|-----------:|
|32.0        |0.545455    |51          |
|10.0        |0.460000    |17          |
|38.0        |0.455285    |45          |
|9.0         |0.451104    |117         |
|22.0        |0.444853    |200         |
|29.0        |0.387013    |123         |
|30.0        |0.379004    |186         |
|5.0         |0.365177    |195         |
|12.0        |0.315789    |4           |
|15.0        |0.250000    |1           |

The content of the most toxic clusters were explored in deep. You can
see some of their content in the following tables:

<details>
  <summary>Cluster #32</summary>

```python
bow_group.get_group(32)[['match','text']]
```

|   match | text                                                |
|--------:|:----------------------------------------------------|
|     183 | you suck at dota                                    |
|     227 | come suck my dick                                   |
|     972 | can you suck my dick?                               |
|     994 | suck dazzle dick after game for boost?              |
|    1216 | u already suck kid                                  |
|    1327 | fucking monkey suck my dick                         |
|    1383 | Sry clock but u suck :)                             |
|    1395 | suck my big black dick                              |
|    1395 | suck my big black cock                              |
|    1459 | and u suck as an orgy                               |
|    1494 | if voker carries we all suck his dick               |
|    1711 | damn slark you suck                                 |
|    2063 | and suck my dick                                    |
|    2201 | suck my balls mirana                                |
|    2459 | you both suck at english                            |
|    2459 | u sure suck at carrying your team                   |
|    2459 | you sure suck your daddy's dick                     |
|    2586 | do u rlly suck that bad at dota like                |
|    2586 | how long did you suck their dick for to boost you?? |
|    2663 | my team suck hard                                   |
|    2869 | you can suck my dick.                               |
|    3017 | you will suck next game                             |
|    3108 | xDD you suck                                        |
|    3303 | ofc suck my dick                                    |
|    3549 | yeah u suck so much u lose to legend xD             |
|    3587 | suck my dick as a present for me                    |
|    3587 | i got no gurl to suck it for me                     |
|    3688 | these guys suck                                     |
|    3792 | suck my dick troll                                  |
|    3824 | suck a dick orange                                  |
|    3869 | what? you wat did you say? suck dick?               |
|    3896 | suck my dick spec                                   |
|    4069 | you must really suck                                |
|    4096 | guys suck dick                                      |
|    4155 | suck my dick riki                                   |
|    4247 | both our teams suck balls                           |
|    4284 | you suck at husk                                    |
|    4284 | yea but you suck at ck                              |
|    4304 | Suck your dick                                      |
|    4304 | ill watch ur mom suck my dick                       |
|    4397 | who has nice tits and wanna suck my cock?           |
|    4586 | suck your momma's dick BOYYE                        |
|    4667 | tinker delete dota u suck                           |
|    4722 | atleats u have a mid who suck                       |
|    4769 | you all suck tho xD                                 |
|    4971 | you cant belive dick my suck                        |
|    4975 | suck my dick you lesser..                           |
|    5028 | suck balanced hgero                                 |
|    5090 | of course suck as can                               |
|    5202 | ur teamates good...u suck                           |
|    5202 | u suck....ur teamate good                           |

</details>

<details>
  <summary>Cluster #10</summary>

```python
bow_group.get_group(10)[['match','text']
```

|   match | text                                                                                                       |
|--------:|:-----------------------------------------------------------------------------------------------------------|
|     183 | you dumb and blind?                                                                                        |
|     183 | its pretty dumb you're complaining, you're literally doing the worst on your team and holding them back LC |
|     290 | my lanes were dumb                                                                                         |
|     537 | that was dumb                                                                                              |
|     539 | real tired of ur dumb ass                                                                                  |
|     741 | cause you guys so dumb cant win mmr                                                                        |
|     839 | sb dumb charge mid too                                                                                     |
|    1381 | talent, ignore this dumb rubick                                                                            |
|    2361 | youre fucking dumb                                                                                         |
|    2361 | you. are. fucking. dumb                                                                                    |
|    2420 | it never was you dumb shit                                                                                 |
|    3276 | can be that dumb                                                                                           |
|    3891 | What a dumb legend player                                                                                  |
|    4189 | the axe pick was dumb... but going 10 death...                                                             |
|    4563 | dumb as you deads ka talaga                                                                                |
|    4902 | are you dumb                                                                                               |
|    4983 | report pudge sp dumb                                                                                       |

</details>

<details>
  <summary>Cluster #38</summary>

```python
bow_group.get_group(38)[['match','text']
```

|   match | text                                                     |
|--------:|:---------------------------------------------------------|
|      83 | stop pause  idiot                                        |
|     389 | i have now SB im not idiot anymore                       |
|     445 | nice mirana idiot                                        |
|     491 | have idiot miranma                                       |
|     554 | fucking idiot offlaner                                   |
|     565 | crystals = skeleton farming idiot                        |
|     775 | keep focus idiot                                         |
|    1437 | sorry my friend is an idiot :)                           |
|    1497 | ok u are idiot too                                       |
|    1711 | you speak idiot?                                         |
|    1711 | i cant do anything idiot gay hero                        |
|    1711 | i cant do anything idiot gay hero                        |
|    1807 | ้he bought sliver edge idiot                              |
|    1807 | idiot your sup                                           |
|    1831 | ez to say when you dont have idiot team mates            |
|    2420 | report this idiot slark                                  |
|    2484 | fucking idiot mid                                        |
|    2623 | fucking idiot team                                       |
|    2726 | alright youre just an idiot                              |
|    2778 | just w8 idiot s shit remack                              |
|    2875 | pick idiot heros                                         |
|    2965 | idiot want pause he is dead                              |
|    3248 | die like a fking idiot                                   |
|    3527 | this idiot mid unpausing                                 |
|    3527 | he wants ez win coz hes too idiot                        |
|    3653 | go turbo mode idiot                                      |
|    3660 | this russian idiot jakiro                                |
|    3701 | see ur tower idiot                                       |
|    3735 | off cheats idiot                                         |
|    3825 | this fuckign idiot team                                  |
|    3896 | just now fucking idiot naix and ck  say me will fail mid |
|    4063 | you hatme me idiot                                       |
|    4096 | Idiot sopt pausing                                       |
|    4096 | idiot never learn                                        |
|    4096 | idiot never learn                                        |
|    4188 | he is the idiot who cant play other role then carry      |
|    4261 | gank me more idiot                                       |
|    4350 | dumb idiot carry                                         |
|    4837 | its sd mode are u idiot?                                 |
|    5074 | idiot invoker ever                                       |
|    5074 | injoker no skill idiot                                   |
|    5158 | idiot why go back]                                       |
|    5202 | of course its sniper idiot                               |
|    5375 | Are you some kind of idiot?                              |
|    5396 | you are fucking idiot                                    |

</details>

A dimensionality reduction with TSNE was applied to visualize the clusters and
highlight the top toxic clusters of the [table](#bow-score-table):

![TSNE graph - clustering](./images/TSNE_bow.png)

##### Results
---

Although the content of the clusters is clearly toxic, its hard to distinguish
subtypes of toxicity in them.

By the nature of this approach and the kind of vectorization applied, you can
recognize that the clustering is too tied to some words, specially those
included in the list of bad words, and that some underlying structures to toxicity
might be hidden or even ignored behind these strong indicators because of their
relevance for the clustering technique applied.

</details>

<details>
  <summary>Word2Vec</summary>

#### Word2Vec clustering
---
To explore further go the the [notebook](./clustering_w2v.ipynb).

The number of clusters choosen for this approach was 40, because any bigger or
lower number didn't improve the distribution of the messages in the clusters.

The distribution of chats in clusters was the following:

![Cluster distribution - Word2Vec](./images/clusters_w2v.png)

As with the bag of words clustering, there are some clusters that include a big
proportion of the messages. Those are clusters that reunites all those messages
that are unique or couldn't be differentiated clearly of the rest of them.
So they were discarded for the current analyisis.

Each message was annotated with their cluster, and then the reduced dataset
was grouped by cluster:

```python
w2v_group = df_test.groupby('w2v_clusters')
```

For each cluster a toxicity score was computed:

```python
w2v_score = (w2v_group['toxicity'].sum() / w2v_group['nwords'].sum())
w2v_scored = pd.DataFrame({'score': w2v_score.values, 'size': w2v_group.size()})
```
Then, they were sorted and filtered to get the cluster with the greatest score:

```python
w2v_scored = w2v_scored[w2v_scored['score'] > 0.5]
w2v_scored = w2v_scored.sort_values(by=['score', 'size'], ascending=[False, False])
```

<a name="w2v-score-table" />
The following results were obtained from the previous code:

|   w2v_clusters |    score |   size |
|---------------:|---------:|-------:|
|             26 | 0.769231 |      5 |
|             23 | 0.611111 |     10 |
|             19 | 0.590164 |     34 |
|             12 | 0.564286 |     71 |
|             22 | 0.557692 |     43 |
|             35 | 0.556818 |     23 |
|             30 | 0.534884 |     11 |
|             28 | 0.533333 |      4 |

The content of the most toxic clusters were explored in deep. You can
see some of their content in the following tables:

<details>
  <summary>Cluster #26</summary>

```python
print(w2v_group.get_group(26)[['match','text']].to_markdown())
```

|   match | text                    |
|--------:|:------------------------|
|      87 | fuck this shit nap team |
|    1785 | fuck that shit          |
|    2931 | fuck off russians shit  |
|    2991 | fuck this shit          |
|    3864 | fuck this shit          |

</details>

<details>
  <summary>Cluster #23</summary>

```python
print(w2v_group.get_group(23)[['match','text']].to_markdown())
```

|   match | text                                                   |
|--------:|:-------------------------------------------------------|
|     636 | so fucking noisy but just a piece of shit              |
|    1421 | fucking lucky shit                                     |
|    1795 | fucking shit ass lion                                  |
|    1988 | we did evry fucking shit for him                       |
|    3662 | fucking trash russian shit                             |
|    3887 | 3 fucking piece of shit                                |
|    3974 | fucking monkey shit                                    |
|    4234 | fucking piece of shit                                  |
|    4253 | you play 2 mid and call us try hard you fucking shit ? |
|    4672 | fucking retarded shit                                  |

</details>

<details>
  <summary>Cluster #19</summary>

```python
print(w2v_group.get_group(19)[['match','text']].to_markdown())
```

|   match | text                                              |
|--------:|:--------------------------------------------------|
|     129 | but my teammates wont to kill u((9(               |
|     636 | u go kill then quit?                              |
|     671 | and u didnt kill me its necro                     |
|     779 | always kill supp                                  |
|    1002 | I won0't kill you                                 |
|    1153 | can only can kill me with ur 2 bodyguards         |
|    1249 | you kill top[                                     |
|    1327 | kill this monkey                                  |
|    1345 | how i kill all of you                             |
|    1395 | or again kill you                                 |
|    1405 | unpause and kill                                  |
|    1751 | Kill urself wywern picker                         |
|    1877 | didnt even get the kill                           |
|    1955 | gj. I can't kill you all                          |
|    2010 | if you kill me its racist                         |
|    2015 | he kill himself 4 minu ago                        |
|    2347 | just go kill him                                  |
|    2659 | u kill him so much                                |
|    2743 | so u wont kill me                                 |
|    2876 | goo kill him I will not save him                  |
|    3095 | see cant even kill zeys                           |
|    3299 | did you level off that kill                       |
|    3823 | for a triple light kill                           |
|    4140 | can i get an aghnims? so your es can't kill me ck |
|    4284 | Because you didnt kill me                         |
|    4301 | you should kill me                                |
|    4304 | i kill you pa and mag whatch me                   |
|    4661 | and you coulkn't even kill me                     |
|    4677 | why you kill                                      |
|    4880 | when i had chancse to kill                        |
|    4936 | kill meif u can bitches                           |
|    5004 | u cant kill me                                    |
|    5202 | yaa For me to kill?                               |
|    5333 | i will kill myself                                |

</details>

<details>
  <summary>Cluster #12</summary>

```python
print(w2v_group.get_group(12)[['match','text']].to_markdown())
```

|   match | text                                    |
|--------:|:----------------------------------------|
|      82 | Ima out to fuck                         |
|     107 | i fuck your mum                         |
|     121 | what the fuck                           |
|     121 | what the fuck                           |
|     153 | where the fuck is snow                  |
|     227 | fuck you buyer                          |
|     378 | cant fuck with oracle!                  |
|     612 | get the fuck out                        |
|     826 | fuck off dudes                          |
|    1033 | bc fuck u thats y                       |
|    1122 | what the fuck                           |
|    1126 | how the fuck is this balance ?          |
|    1256 | chrono me for what u fuck               |
|    1260 | fuck your moms                          |
|    1274 | fuck that necro he sucks anyways        |
|    1437 | waif fuck your beloved oness            |
|    1741 | can you fuck off ?                      |
|    1859 | fuck that dark willow                   |
|    1877 | alright, fuck you kunkka                |
|    1877 | again.. fuck you kunkka                 |
|    1890 | what the fuck?                          |
|    1898 | 17minutes fuck off rd??                 |
|    1945 | fuck your family                        |
|    1991 | how the fuck                            |
|    2015 | go fuck off somewhere else              |
|    2044 | will fuck you again bird                |
|    2083 | what the fuck                           |
|    2174 | fuck me when i  save this moron         |
|    2201 | the fuck is nam nam                     |
|    2328 | the fuck are you doing                  |
|    2361 | fuck off already                        |
|    2436 | why the fuck u tped                     |
|    2440 | fuck all the niggas'                    |
|    2458 | i did not fuck your gf                  |
|    2548 | go fuck yourself bick                   |
|    2596 | can u call him a fat fuck for me?       |
|    2646 | idk how the fuck he feeds so much       |
|    2768 | the fuck  you doing there               |
|    2803 | fuck you all                            |
|    2931 | fuck off and act mature                 |
|    2931 | fuck you all                            |
|    3204 | fuck you rat                            |
|    3302 | fuck outta here                         |
|    3535 | fuck this cxarry                        |
|    3599 | i have places to be and bitches to fuck |
|    3622 | saved you lil' fuck                     |
|    3722 | fuck off brah                           |
|    3762 | fuck you  kotl                          |
|    3762 | fuck you kotl                           |
|    4071 | fuck off weab                           |
|    4088 | fuck the courier                        |
|    4164 | fuck you bug                            |
|    4228 | Fuck you Wes POS                        |
|    4304 | i only fuck muslim girls                |
|    4304 | i fuck your sister magnus               |
|    4304 | i only fuck muslim girls                |
|    4304 | i fuck gigi hadid                       |
|    4355 | fuck you heart                          |
|    4541 | fuck rsa unranked                       |
|    4702 | the fuck are you doing                  |
|    4813 | i did fuck your mum                     |
|    4820 | fuck you nimnko                         |
|    4844 | who the fuck cares                      |
|    4962 | who can fuck me？                       |
|    5030 | i will fuck u too now                   |
|    5095 | fuck the kids                           |
|    5240 | fuck key board                          |
|    5250 | wake the fuck up                        |
|    5295 | fuck you nigg                           |
|    5336 | fuck off tusk                           |
|    5395 | fuck you sniepr                         |

</details>

<details>
  <summary>Cluster #22</summary>

```python
print(w2v_group.get_group(22)[['match', 'text']].to_markdown())
```

|   match | text                                                                                                            |
|--------:|:----------------------------------------------------------------------------------------------------------------|
|     111 | fucking hard game ever when got lucky timing skill on enemy                                                     |
|     121 | ur mom is fucking slut                                                                                          |
|     121 | are fucking retard ursa                                                                                         |
|     290 | this lineup is fucking stupid                                                                                   |
|     339 | retard my fucking team                                                                                          |
|     554 | fucking idiot offlaner                                                                                          |
|     598 | report necro and fucking riki                                                                                   |
|     683 | jugg you fucking moron                                                                                          |
|     702 | go fucking kill him                                                                                             |
|    1125 | Already pick first invoker and i playing my role. Picking fucking ck and go mid againist tinker what an asshole |
|    1269 | fucking retarded russian dogs                                                                                   |
|    1327 | fucking monkey suck my dick                                                                                     |
|    1353 | fucking asshole mid more                                                                                        |
|    1364 | this pudge so fucking stupid                                                                                    |
|    1407 | this invo so fucking asshole                                                                                    |
|    1540 | u fucking selfish prick qop                                                                                     |
|    1621 | you fucking suck wk                                                                                             |
|    1710 | fucking pussy team man                                                                                          |
|    1890 | fucking try harder                                                                                              |
|    2343 | ? ur a bristle you fucking retard                                                                               |
|    2484 | fucking idiot mid                                                                                               |
|    2623 | fucking idiot team                                                                                              |
|    2917 | it is 5v5 game u fucking cunt                                                                                   |
|    3018 | i fucking kill you                                                                                              |
|    3177 | end this is bulshit gaming fucking sniper and pudge watching porn                                               |
|    3188 | fucking clueless fuck                                                                                           |
|    3422 | why u fucking hounting kills just push                                                                          |
|    3629 | fucking kill yourself                                                                                           |
|    3725 | fucking god can't give less than 4 secs                                                                         |
|    3748 | team is fucking stupid animals                                                                                  |
|    3894 | why i have this fucking retarded russian in my team                                                             |
|    3896 | just now fucking idiot naix and ck  say me will fail mid                                                        |
|    3911 | it is fucking unranked chill the fuck out                                                                       |
|    4255 | dude if you're gonna bitch this much then dont fucking defedn stop being a punk                                 |
|    4265 | CAmila fucking bitch                                                                                            |
|    4304 | shut up u fucking virign cunt                                                                                   |
|    4304 | fucking indian fuck                                                                                             |
|    4366 | Fucking Russians retard                                                                                         |
|    4640 | just fucking kill me                                                                                            |
|    4754 | that is fucking bitch                                                                                           |
|    5216 | easy fucking slave                                                                                              |
|    5241 | that fucking ant is dead so im happy                                                                            |
|    5396 | you are fucking idiot                                                                                           |

</details>

A dimensionality reduction with TSNE was applied to visualize the clusters and
highlight the top toxic clusters of the [table](#w2v-score-table):
![TSNE graph - clustering](./images/TSNE_w2v.png)
 
##### Results
---
It is important to note first the drawbacks of the score designed to reduce the
scope of our analysis. Cluster number 19, for example, is not predominantly toxic
but it has a higher score than clusters 12 and 22, that are clearly toxic. This is
due to the nature of game, where the word "kill" is a common word.

That said, it can be seen that the improvements in relation with the bag of 
words approach are not so great. The most toxic clusters are smaller, the
messages in them have more words in common and some syntax structure relation
can be recognized. But all of that is not enough to obtain clearly separated kinds
of toxicity in them.

</details>

<details>
  <summary>FastText</summary>

#### FastText clustering
---
To explore further go the the [notebook](./clustering_ftt.ipynb).

The number of clusters choosen for this approach was 100, because it generated
a well distributed clustering with a good number of messages in each cluster.

The distribution of chats in clusters was the following:

![Cluster distribution - FastText](./images/clusters_ftt.png)

The distribution shown above is very close to uniformity, although some
outerliers my indicate lack of cohesion in the cluster.

Each message was annotated with their cluster, and then the reduced dataset
was grouped by cluster:

```python
ftt_group = df_test.groupby('ftt_clusters')
```

For each cluster a toxicity score was computed:

```python
ftt_score = (ftt_group['toxicity'].sum() / ftt_group['nwords'].sum())
ftt_scored = pd.DataFrame({'score': ftt_score.values, 'size': ftt_group.size()})
```
Then, they were sorted and filtered to get the cluster with the greatest score:

```python
ftt_scored = ftt_scored[ftt_scored['score'] > 0.25]
ftt_scored = ftt_scored.sort_values(by=['score', 'size'], ascending=[False, False])
```

<a name="ftt-score-table" />
The following results were obtained from the previous code:

|   ftt_clusters |    score |   size |
|---------------:|---------:|-------:|
|             55 | 0.631579 |     19 |
|             33 | 0.627119 |     39 |
|             80 | 0.625    |     22 |
|              2 | 0.391813 |     76 |
|             68 | 0.354167 |     67 |
|             43 | 0.317757 |     44 |
|             25 | 0.270588 |     73 |
|             41 | 0.260771 |    133 |

The content of the most toxic clusters were explored in deep. You can
see some of their content in the following tables:

<details>
  <summary>Cluster #55</summary>

```python
ftt_group.get_group(55)[['match','text']]
```

|   match | text                    |
|--------:|:------------------------|
|     121 | you are shit            |
|    1249 | had to take a shit      |
|    1504 | name of the ship        |
|    1552 | thats shit              |
|    1797 | youre shit              |
|    1797 | why is ur name shit     |
|    1906 | and cant do shit        |
|    2347 | i have the shittiest    |
|    2793 | with this shitshiow     |
|    2975 | i dont give shit        |
|    3177 | 1-10 shitt              |
|    3662 | and ds cant do shit     |
|    4332 | with this shit          |
|    4479 | i wont hit u            |
|    4722 | hit him more            |
|    4736 | cant even lashit        |
|    4837 | what now mk and bh shit |
|    4902 | then do this shit       |
|    4963 | i cant do shit          |

</details>

<details>
  <summary>Cluster #33</summary>

```python
ftt_group.get_group(33)[['match','text']]
```

|   match | text                             |
|--------:|:---------------------------------|
|      82 | Ima out to fuck                  |
|     121 | what the fuck                    |
|     121 | what the fuck                    |
|     368 | It fucked me more than you.      |
|     612 | get the fuck out                 |
|     796 | what the fukc                    |
|     823 | willow can't admit she fucked u  |
|     981 | what the fucl                    |
|    1033 | bc fuck u thats y                |
|    1122 | what the fuck                    |
|    1269 | WHY THE FUCK ARE YOU THERe       |
|    1345 | she almost fucked me up          |
|    1741 | can you fuck off ?               |
|    1859 | fuck that dark willow            |
|    1877 | alright, fuck you kunkka         |
|    1890 | what the fuck?                   |
|    1898 | 17minutes fuck off rd??          |
|    1991 | how the fuck                     |
|    2015 | go fuck off somewhere else       |
|    2083 | what the fuck                    |
|    2328 | the fuck are you doing           |
|    2361 | fuck off already                 |
|    2458 | i did not fuck your gf           |
|    2768 | the fuck  you doing there        |
|    2803 | fuck you all                     |
|    2931 | fuck you all                     |
|    3750 | go to russian servers u fuck     |
|    3777 | fuck off back to russian servers |
|    4100 | i will skullfuck you             |
|    4102 | WHAT THe FUCKIN PC               |
|    4296 | FUCK THIS IDI0TS                 |
|    4304 | i only fuck muslim girls         |
|    4304 | i only fuck muslim girls         |
|    4304 | i fuck gigi hadid                |
|    4702 | the fuck are you doing           |
|    4901 | fuck you tinker picker           |
|    4960 | go fuck urself man               |
|    4962 | who can fuck me？                |
|    5030 | i will fuck u too now            |

</details>

<details>
  <summary>Cluster #80</summary>

```python
ftt_group.get_group(80)[['match','text']]
```

|   match | text                                           |
|--------:|:-----------------------------------------------|
|      71 | I WOULD HAVE KILL HIM                          |
|    1002 | I won0't kill you                              |
|    1249 | you kill top[                                  |
|    1345 | how i kill all of you                          |
|    1395 | or again kill you                              |
|    1877 | didnt even get the kill                        |
|    1955 | gj. I can't kill you all                       |
|    2347 | just go kill him                               |
|    2586 | 8k mmr can'ty even get a kill in these lobbies |
|    2659 | u kill him so much                             |
|    2743 | so u wont kill me                              |
|    2918 | not your 'skill'                               |
|    3463 | not same skill                                 |
|    3463 | not same skill                                 |
|    3501 | now that was not a skill                       |
|    3740 | its your skill                                 |
|    4284 | Because you didnt kill me                      |
|    4301 | you should kill me                             |
|    4677 | why you kill                                   |
|    4983 | IT WILL TAKE 5 TO KILL NE                      |
|    5004 | u cant kill me                                 |
|    5333 | i will kill myself                             |

</details>

<details>
  <summary>Cluster #20</summary>

```python
ftt_group.get_group(2)[['match','text']]
```

|   match | text                                                          |
|--------:|:--------------------------------------------------------------|
|      71 | FUCK THIS PANGO                                               |
|     107 | i fuck your mum                                               |
|     153 | where the fuck is snow                                        |
|     235 | SHUT THE FUCK UP                                              |
|     378 | cant fuck with oracle!                                        |
|     678 | fuck your mom                                                 |
|     736 | who ther fuck gets bkb in LP?                                 |
|     826 | fuck off dudes                                                |
|     915 | what the fuck is your problem with m?                         |
|     916 | she gay and she fucked u                                      |
|    1003 | fuck your mother                                              |
|    1093 | viper accountbuyer 100%                                       |
|    1126 | how the fuck is this balance ?                                |
|    1244 | one moment silence please                                     |
|    1256 | chrono me for what u fuck                                     |
|    1260 | fuck your moms                                                |
|    1283 | YOU fuck only your mother                                     |
|    1437 | waif fuck your beloved oness                                  |
|    1540 | all is my fuckign intenrt                                     |
|    1871 | SHUT THE$ FUCK UP                                             |
|    1877 | again.. fuck you kunkka                                       |
|    1882 | these mother Fuckers                                          |
|    1933 | puck I will FUCK YOU                                          |
|    1945 | fuck your family                                              |
|    2033 | fuck that kid                                                 |
|    2044 | will fuck you again bird                                      |
|    2092 | get fucked later                                              |
|    2420 | fucken lvl 16 after 44min?                                    |
|    2436 | why the fuck u tped                                           |
|    2440 | fuck all the niggas'                                          |
|    2548 | go fuck yourself bick                                         |
|    2596 | can u call him a fat fuck for me?                             |
|    2655 | 100% mther fuker                                              |
|    2738 | if i was not in this voids team i wish he has a loss  fuck it |
|    2793 | i had to turn on music...                                     |
|    2989 | fuck you techies                                              |
|    3011 | just endla fucker                                             |
|    3179 | so hard take 3 hero XD                                        |
|    3204 | fuck you rat                                                  |
|    3302 | fuck outta here                                               |
|    3325 | see mother fucker                                             |
|    3535 | fuck this cxarry                                              |
|    3554 | silence now SNIPER                                            |
|    3556 | CAN U JUST SHUT THE FUCK UP?                                  |
|    3574 | omni i would fuck that girl on your pic                       |
|    3701 | FUCK THIS TECHIES                                             |
|    3762 | fuck you  kotl                                                |
|    3762 | fuck you kotl                                                 |
|    3869 | jugg fuck you                                                 |
|    3896 | fuck you fuck you fuck you fuck you                           |
|    3911 | it is fucking unranked chill the fuck out                     |
|    4071 | fuck off weab                                                 |
|    4074 | shut the hell up                                              |
|    4084 | i will fuck your mother                                       |
|    4123 | can u shut the fuck up                                        |
|    4304 | u never fucked a girl havent u                                |
|    4304 | ill fuck her                                                  |
|    4304 | lets see ill fuck am's mom                                    |
|    4304 | after i fuck ur mom                                           |
|    4316 | fuck you mirana                                               |
|    4355 | fuck you heart                                                |
|    4493 | fuck you friend                                               |
|    4564 | fuck u are bad                                                |
|    4745 | this fuckgin tea                                              |
|    4813 | i did fuck your mum                                           |
|    4820 | fuck you nimnko                                               |
|    4844 | who the fuck cares                                            |
|    5109 | and said FUCK YOU                                             |
|    5135 | FUCK your mom                                                 |
|    5245 | am delete dota and seek help u mental fuck                    |
|    5286 | this is our trash fuck lc                                     |
|    5295 | fuck you nigg                                                 |
|    5333 | ok kill yourself u fuck                                       |
|    5336 | fuck off tusk                                                 |
|    5363 | fuck your mom 2on1                                            |
|    5395 | fuck you sniepr                                               |

</details>

A dimensionality reduction with TSNE was applied to visualize the clusters and
highlight the top toxic clusters of the [table](#ftt-score-table):

![TSNE graph - clustering](./images/TSNE_ftt.png)

##### Results
---

The hipothesis about the use of FastText was that the usage of character
embeddings will improve the quality of clusters because it will allow the
capture of slang vocabulary or abbreviations.

That hipothesis show to be wrong at the moment of analysing the clusters.
Although the clusters displayed contain a high number of toxic messages, they
are blended with non-toxic messages related to the development of the game.

However, someone may argue that the proportion of toxic messages is great. That
impresion can be misleading because the proportion is related to the fact that
the clusters were filtered and selected watching the score of toxicity of each
one and that number can have great variations when using larger datasets or
different numbers of clusters.

Finally, the characteristics mentioned above imply that it couldn't be found any
topic or type of toxicity that gather all the messages in a cluster or at least,
a great number of them.

</details>

</details>

<details>
  <summary>Conclusion</summary>

### Conclusion

Although the objective of this project couldn't be meet, it produced some
conclusions that are worth mentioning:

- The proliferation of toxicity in this dataset was clear and the intensity of
  it is high.
- In word2vec and bag of words the clusters can be clearly disinguished by the
  proliferation of some toxic words in the content of their messages. In
  FastText however, clustering fails to separate toxic messages from messages
  who are not.
- Clustering seems to lack the abilty to abstract elements that contribute to
  the toxicity of a message beyond the toxic words in it.
- The clusters obtained from the bag of words and word2vec clusterings can be
  used to train a binary clasification model to detect toxicity in game chats.
- A well designed report system in online chats could provide a free source of
  annotated data to use in the analysis of toxicity and in the development of
  hybrid models (human - machine) for moderation.

</details>

