import subprocess
import sys
import streamlit as st
import random
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def get_id_from_info(song, artist, info):
    song_entry = info[(info['song'] == song) & (info['artist'] == artist)]
    
    if not song_entry.empty:
        id = song_entry.iloc[0]['id']
        return id
    
def cos_sim(arr1, arr2):
    arr1_reshape = arr1.reshape(1, -1)
    arr2_reshape = arr2.reshape(1, -1)
    res = cosine_similarity(arr1_reshape, arr2_reshape)[0][0]
    return res

def audio_based(id, dfrepr, N, sim_func):
    # return the query song's row in repr
    target_row = dfrepr[dfrepr['id'] == id].iloc[:, 2:].to_numpy()
    # calculate similarity score
    dfrepr['sim_score'] = dfrepr.apply(lambda x:sim_func(x[2:].to_numpy(),target_row), axis=1)
    # sort tracks by similarity 
    sorted_repr = dfrepr.sort_values(by='sim_score', ascending=False)
    # get the N most similar tracks 
    res = sorted_repr.iloc[1: N+1]['id'].to_numpy()
    return res 

def get_genre(id,genres_df):
  # print(genres_df[genres_df['id'] == id ]['id'].values[0],'--->',id)
  return set(genres_df[genres_df['id'] == id ]['genre'].values[0].replace("[", "").replace("]", "").replace("'", "").split(', '))

# getting the genres of the retrieved ids
#get_genre(id,genres_df)
def id_and_url_or_genre(query_id,retrieved_ids,df,func):
  retrieved_g_u = []
  for id in retrieved_ids:
    retrieved_g_u.append(func(id,df))

  query_g_u = func(query_id,df)
  print(f'Query id: {query_id} ---- Query genre_or_url: {query_g_u}')
  print('---------------------------------------------------------------------------')

  for id, ret in(zip(retrieved_ids,retrieved_g_u)):
    print(f'Retrieved id: {id} ---- Retrieved genre_or_url: {ret}')
  return query_g_u, retrieved_g_u
# query_genre, retrieved_genre = id_and_url_or_genre(query_id=query_id,retrieved_ids=retrieved_ids,df=genre,func=get_genre)


def url(id,url_df):
  return url_df[url_df['id']==id].values[0]
#####################################################################################################################################################################################


f = pd.read_csv("/id_information_mmsr.tsv", delimiter='\t')
# df_artist = df['artist'].values.tolist()
# df_song = df['song'].values.tolist()

# df['artist_song'] = df['artist'] + ' - ' + df['song']
# df_artist_song = df['artist_song'].values.tolist()


# # df_artist#.values.tolist()

# st.title("Music Retrieval System")
