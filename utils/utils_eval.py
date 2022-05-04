import seaborn as sns
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from math import log
import matplotlib.style
matplotlib.style.use('ggplot')
from random import random


def parser_for_eval(data,dataset) : 
  for filename in pd.unique(data.piano_roll_name): 
    df = data[data.piano_roll_name == filename].reset_index(drop=True)
    cumuldt = 0
    idx=0
    pitchseqs = []
    tseqs = []
    dTseqs = []
    while idx < len(df):
      if len(np.unique(df[df.columns[2:]].loc[idx])) > 1 : 
          pitch_t = np.where(df[df.columns[2:]].loc[idx] != 0)[0][0]
          T = 0
          while True:
              idx += 1
              #print(idx)
              T += 1
              if idx > (len(df) - 1):
                break
              if len(np.unique(df[df.columns[2:]].loc[idx])) <= 1 :
                break
              if np.where(df[df.columns[2:]].loc[idx] != 0)[0][0] != pitch_t :
                break

          if T == 0.:  # Don't consider 0 duration notes
              continue
          idx
          candidateT = T / 4
          if candidateT == 0.:  # note that are maped to 0 duration
              cumuldt += 1
              continue

          pitchseqs.append(pitch_t)
          tseqs.append(candidateT)
          dt = cumuldt/4
          dTseqs.append(dt)
          cumuldt = 0
      else : 
          cumuldt += 1
          idx+=1
          #print(idx)
    dataset['pitchseqs'].append(pitchseqs)
    dataset['tseqs'].append(tseqs)
    dataset['dTseqs'].append(dTseqs)
  return(dataset)

def getSong(dataset, i):
    """
    return song i of the dataset
    """
    return {"dTseqs": dataset["dTseqs"][i],
            "tseqs": dataset["tseqs"][i],
            "pitchseqs": dataset["pitchseqs"][i]}

def getNote(song, i):
    return (song["dTseqs"][i], song["tseqs"][i], song["pitchseqs"][i])

def getLength(song, dictionaries):
    l = 0
    for dT in song["dTseqs"]:
        l += dictionaries["dTseqs"][dT]
    l += dictionaries["tseqs"][song["tseqs"][-1]]
    return l

def normalize(d):
    """
    Returns a normalized version of the dictionary d
    :param d: a dictionary mapping to POSITIVE numbers
    :return: the normalized dictionary
    """
    Z = 0
    res = defaultdict(float)
    for key in d:
        Z += d[key]
    for key in d:
        res[key] = d[key]/float(Z)
    return res

def mydefaultdict():
    return defaultdict(float)

def trainsingleorder(data, order):
    """
    Returns a trained dictionary on data at given order for a single sequence
    :param data: a sequence
    :param order: an int (0 for frequency count, 1 for Markov...)
    :return: if order is 0 a dict {value -> prob of occurrence}
             for bigger orders a dict {str(history) -> {value -> prob of occurrence} }
    """
    if order == 0:
        res = defaultdict(float)
        for song in data:
            for i in song:
                res[i] += 1
        res = normalize(res)
        return res
    else:
        res = defaultdict(mydefaultdict)
        # Counting occurrences of transitions
        for song in data:
            for i in range(len(song) - order):
                hist = str(song[i:i + order])
                n = song[i + order]
                res[hist][n] += 1
        # Normalization
        for hist in res:
            res[hist] = normalize(res[hist])
        return res

def dic_argmax(d):
    maxi = 0
    argmax = None
    for k in d:
        if d[k] > maxi:
            maxi = d[k]
            argmax = k
    return argmax

def dic_sample(d):
    dn = normalize(d)
    u = random()
    cumulative = 0
    elts = sorted(d.keys())
    for k in elts:
        cumulative += dn[k]
        if cumulative > u:
            return k

def keys_subtract(d, x):
    res = defaultdict(float)
    for k in d:
        res[k-x] = d[k]
    return res

def tvDistance(p, q):
    """
    total variation distance
    """
    res = 0.
    for key in set(p.keys()).union(set(q.keys())):
        res += abs(p[key] - q[key])
    return 0.5 * res

def analyze_chords(real_data, gen_data, title="Chord decomposition", real_dis=None, 
                   show_plot=False, plot_fp=None):
    """
    Analysis of intervals between notes in a same chord
    :param real_data: a dataset of the reference corpus (can be None if real_dis is already given)
    :param gen_data: dataset of studied corpus
    :param title: title for the plot
    :param real_dis: if given, the distribution on the reference corpus will not be recomputed
    :param show_plot: True to directly show plot
    :param plot_fp: if not None, where to save the plot
    :returns: total variation distance between the real and the generated distribution
    """
    gen_dis = defaultdict(float)

    if real_dis is None:
        real_dis = defaultdict(float)
        for s, song in enumerate(real_data['dTseqs']):
            cur_chord = set()
            for i, dT in enumerate(song):
                if dT == 0:
                    p = real_data['pitchseqs'][s][i]
                    for x in cur_chord:
                        diff = abs(p-x) % 12
                        real_dis[diff] += 1
                    cur_chord.add(p)
                else:
                    cur_chord = {real_data['pitchseqs'][s][i]}

    for s, song in enumerate(gen_data['dTseqs']):
        cur_chord = set()
        for i, dT in enumerate(song):
            if dT == 0:
                p = gen_data['pitchseqs'][s][i]
                for x in cur_chord:
                    diff = abs(p - x) % 12
                    gen_dis[diff] += 1
                cur_chord.add(p)
            else:
                cur_chord = {gen_data['pitchseqs'][s][i]}
    
    # normalization
    real_dis = normalize(real_dis)
    gen_dis = normalize(gen_dis)

    # Make plot
    fig, ax = plt.subplots()
    df = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [real_dis[i] for i in range(12)],
                       'distribution': 'real'})
    df2 = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [gen_dis[i] for i in range(12)],
                       'distribution': 'generated'})
    df = pd.concat([df, df2])
    sns.barplot(x='intervals', y='frequency', hue='distribution', data=df, ax=ax)
    fig.suptitle(title)

    if show_plot:
        fig.show()
    elif plot_fp is not None:
        plt.savefig(plot_fp)

    # Compute statistical distance
    return tvDistance(real_dis, gen_dis)

def analyze_intervals(real_data, gen_data, title="Interval decomposition", real_dis=None, 
                      show_plot=False, plot_fp=None):
    """
    Analysis of intervals between successive notes
    :param real_data: a dataset of the reference corpus (can be None if real_dis is already given)
    :param gen_data: dataset of studied corpus
    :param title: title for the plot
    :param real_dis: if given, the distribution on the reference corpus will not be recomputed
    :param show_plot: True to directly show plot
    :param plot_fp: if not None, where to save the plot
    :returns: total variation distance between the real and the generated distribution
    """
    gen_dis = defaultdict(float)

    if real_dis is None:
        real_dis = defaultdict(float)
        for s, song in enumerate(real_data['dTseqs']):
            for i, dT in enumerate(song):
                if i > 0:
                    diff = abs(real_data['pitchseqs'][s][i]-p) % 12
                    real_dis[diff] += 1
                p = real_data['pitchseqs'][s][i]

    for s, song in enumerate(gen_data['dTseqs']):
        for i, dT in enumerate(song):
            if i > 0:
                diff = abs(gen_data['pitchseqs'][s][i]-p) % 12
                gen_dis[diff] += 1
            p = gen_data['pitchseqs'][s][i]

    # Normalize
    real_dis = normalize(real_dis)
    gen_dis = normalize(gen_dis)

    # Make plot
    fig, ax = plt.subplots()
    df = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [real_dis[i] for i in range(12)],
                       'distribution': 'real'})
    df2 = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [gen_dis[i] for i in range(12)],
                       'distribution': 'generated'})
    df = pd.concat([df, df2])
    sns.barplot(x='intervals', y='frequency', hue='distribution', data=df, ax=ax)
    fig.suptitle(title)

    if show_plot:
        fig.show()
    elif plot_fp is not None:
        plt.savefig(plot_fp)
    
    # Compute statistical distance
    return tvDistance(real_dis, gen_dis)
