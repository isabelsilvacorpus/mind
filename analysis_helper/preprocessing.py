import numpy as np 
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch

    
def load_embeddings(f_path):
    '''
    Read MIND .vec files, which are structured as ID | 100-d embedding
    '''
    with open(f_path, 'r') as vec_file:
        lines = vec_file.readlines()
    news = []
    vectors = []
    for line in lines: 
        parts = line.strip().split('\t')
        news.append(parts[0])
        vectors.append([float(i) for i in parts[1:]])
    return news, np.array(vectors)

def process_tsv(df, cols): 
    '''
    Read MIND news, behavior .tsv files
    '''
    for col_i in cols:
        df[col_i].fillna('{}', inplace=True)
        df[col_i] = df[col_i].apply(literal_eval)
    return df

def extract_entity_list(df, col, sought_value):
    '''
    Extract entity ID's from pandas df. 
    For MIND news.tsv use "WikidataId" as sought_value
    '''
    entity_id = []
    for i in range(len(df)):
        entity_id.append([d[sought_value] for d in df[col][i] if sought_value in d])
    return entity_id

def normalize(vec):
    '''
    Normalize embeddings 
    '''
    norms = np.apply_along_axis(np.linalg.norm, 1, vec)
    vec = vec / norms.reshape(-1,1)
    return vec

def prepare_tsne_data(df, x_col, y_col):
    X = [x.reshape(100,) for x in df[x_col]]
    y = df[y_col]
    return np.array(X), y

def cat_to_int(df, col_name):
    le = LabelEncoder()
    y = le.fit_transform(df[col_name])
    return y

def ohe(df, col_name):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df[[col_name]])
    y = ohe.transform(df[[col_name]])
    return y

def vecs_to_df(vec_array, vec_id):
    vectors = [vec.reshape(1,-1) for vec in vec_array]
    vec_df = pd.DataFrame(data = {"entity_id": vec_id, "entity_vec": vectors})
    return vec_df

def mean_pooled_news_embeddings(vec_array, vec_id, metadata_df, entity_id_col):
    entity_df = vecs_to_df(vec_array, vec_id)
    metadata_df = metadata_df.explode(entity_id_col)
    metadata_df = metadata_df[metadata_df[entity_id_col].notnull()]
    merged_df = pd.merge(metadata_df, entity_df, right_on="entity_id", left_on=entity_id_col, how = "inner")
    news_vecs = merged_df.groupby(['news_id', 'category'])['entity_vec'].apply(lambda x: np.mean(x, axis=0)).reset_index()
    news_vecs['entity_vec'] = news_vecs.entity_vec.apply(lambda x: x.flatten())
    return news_vecs

def test_training_split(dataset, train_prop):
    train_size = int(train_prop * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset