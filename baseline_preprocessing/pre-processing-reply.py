import numpy as np
from scipy.sparse import csr_matrix
import random
import keras.backend as K
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
import csv
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import networkx as nx

# df = pd.read_csv('data/friend_list.csv',sep="\t")
# g = nx.from_pandas_edgelist(df, "follower", "followee", edge_attr=True)
# adjacent_list = nx.to_dict_of_lists(g, nodelist=None)
# nx.write_adjlist(g,"data/friend.adjlist")

df_label = pd.read_csv('data/origin_data/dict.csv',sep="\t",
        header=0,
		usecols=[1,3],
		names=["twitter_id","party"])
df_label.replace(('R', 'D'), (0, 1), inplace=True)


df = pd.read_csv('data/origin_data/reply_list.csv',sep="\t")
g = nx.from_pandas_edgelist(df, "user_id", "reply_to", edge_attr=True)
df_label = df_label[
					df_label['twitter_id'].apply(lambda x: x in list(g.nodes))
  					]


print(len(g.nodes))
print(len(g.edges))
df_label.to_csv("data/reply_labels_baseline.txt", header = False, columns = ['twitter_id','party'], sep=' ', index = False)
df.to_csv("data/reply_list_baseline.csv", header = False, columns = ['user_id','reply_to'], sep=' ', index = False)