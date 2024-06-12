"""
Jeffrey Shen - 05/27/2024
CIS 600 - Syracuse University
Video Game Recommender Project - Cosine Similarity vs Euclidean Distance

This program builds 2 video game recommender systems to compare the effectiveness of each similarity technique.
The data set used is the PC Steam catalog from kaggle https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data

The program loads the dataset CSV file into a pandas dataframe (85103 x 39 as of 01/08/2024)
Some preprocessing is then performed: 
- Lowercase all text
- Remove any datapoints with zero recommendations (13760 x 39)
- Remove any irrelevant columns, keeping only: ["name", "release date", "about the game", "supported languages", "windows", "categories", "genres", "tags"]
    (13760 x 8)
- Filter with the following:
    - windows == True
    - supported languages contains English
    - remove non-alphanumeric symbols
    (13356)
- Remove any null points for about the game (13331)
- Combine to a feature column and then drop: about the game, genres, categories, tags (13331 x 3)

The feature column is then tokenized using CountVectorizer. 
Then fit_transform is applied:
- Fit: learns vocab and builds a dict of unique words
- Transform: converts text data to numerical vectors, each unique word is a feature and value is count
(68017 unique words, each datapoint has vector of size 68017, thus 68017 dimensions)

Finally we build 2 models using the vectorized data:
Cosine Similarity and Euclidean Distance
    

"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def load_data(dataset="games_dataset.csv"):
    games_df = pd.read_csv(dataset)
    return games_df


def preprocess_data(df):
    # Lowercase everything
    df.columns = df.columns.str.lower()
    df = df.map(lambda s: s.lower() if type(s) == str else s)

    # Remove datapoints with 0 recommendations
    df = df[df["recommendations"] > 0]
    
    # Remove irrelevant columns
    keep_col = ["name", "release date", "about the game", "supported languages", "windows", "categories", "genres", "tags"]
    df = df.copy()
    for col in df.columns:
        if col not in keep_col:
            df.drop(columns=[col], inplace=True)
            
    # Remove non-Windows, non-English, and non-Alphanumeric char's
    df = df[df["windows"] == True]
    df = df[df["supported languages"].apply(lambda x: "english" in x)]
    df["name"] = df["name"].str.replace(r"[^\w\s']+", '', regex=True)    
    
    # Remove null datapoints for about the game column
    df = df.dropna(subset=["about the game"])
    
    # Consolidate feature tags to feature column
    df["feature"] = df["about the game"] + df["genres"] + df["categories"] + df["tags"]
    df = df.drop(columns=["about the game", "genres", "categories", "tags", "windows", "supported languages"])
    
    df.reset_index(drop=True, inplace=True)
    return df


def vectorize_data(df):
    cv = CountVectorizer()
    vector = cv.fit_transform(df["feature"].values.astype("U")).toarray()    
    return vector


def cos_sim(vector):
    similiarity = cosine_similarity(vector)
    return similiarity

    
def eucl_dist(vector):
    distance = euclidean_distances(vector)
    return distance
    

def recommend_sim(game, df, similarity):
    index = df[df["name"] == game].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    cnt = 1
    for i in distance[0:11]:
        if i[0] == index:
            print(f"Top 10 games similar to: {df.iloc[i[0]]["name"]}")
            continue
        print(f"[{cnt}]:\t{df.iloc[i[0]]["name"]}")
        cnt += 1


def recommend_dist(game, df, distance):
    index = df[df["name"] == game].index[0]
    distances = list(enumerate(distance[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1])
    cnt = 1
    for i in sorted_distances[0:11]:  
        if i[0] == index:
            print(f"Top 10 games similar to: {df.iloc[i[0]]["name"]}")   
            continue     
        print(f"[{cnt}]:\t{df.iloc[i[0]]["name"]}")
        # print(games_df.iloc[i[0]]["name"])
        cnt += 1


def recommend(game, df, similarity, distance):
    index = df[df["name"] == game].index[0]
    
    # Get similarity recommendations
    sim_distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    sim_recommendations = [df.iloc[i[0]]["name"] for i in sim_distance[1:11]]  # Exclude the first one because it's the input game
    
    # Get distance recommendations
    dist_distances = sorted(list(enumerate(distance[index])), key=lambda x: x[1])
    dist_recommendations = [df.iloc[i[0]]["name"] for i in dist_distances[1:11]]  # Exclude the first one because it's the input game
    
    # Combine into a DataFrame
    recommendations_df = pd.DataFrame({
        'COSINE SIMILARITY RECOMMENDATIONS': sim_recommendations,
        'EUCLIDEAN DISTANCE RECOMMENDATIONS': dist_recommendations
    })
    
    return recommendations_df


def main():
    print("Loading data, please wait...")
    start_time = time.time()
    df = load_data()
    print(f"Data loaded! [{round(time.time() - start_time, 2)} sec]")
    
    print("\nPreprocessing data, please wait...")
    start_time = time.time()
    df = preprocess_data(df)
    print(f"Preprocessing finished! [{round(time.time() - start_time, 2)} sec]")
    
    print("\nVectorizing data, please wait...")
    start_time = time.time()
    vector = vectorize_data(df)
    print(f"Vectorizing finished! [{round(time.time() - start_time, 2)} sec]")
    
    print("\nCalculating Cosine similarities, please wait...")
    start_time = time.time()
    sim = cos_sim(vector)
    print(f"Cosine similarity finished! [{round(time.time() - start_time, 2)} sec]")
    
    print("\nCalculating Euclidean distance, please wait...")
    start_time = time.time()
    dist = eucl_dist(vector)
    print(f"Euclidean distance finished! [{round(time.time() - start_time, 2)} sec]")
    
    while True:
        game = input("\nEnter a game or X to exit: ")
        # recommend_sim(game, df, sim)
        # recommend_dist(game, df, dist)
        if game.lower() in ["x"]:
            exit()
        try:
            print(recommend(game, df, sim, dist))
        except Exception as e:
            print("Search failed! Please check your spelling!\n DEBUG >>> Exception: ", e)
    


if __name__ == "__main__":
    main()
