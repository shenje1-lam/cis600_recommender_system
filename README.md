# CIS600 - Data/Knowledge Mining Project

## Installation
- This project was developed with Python 3.12.1
- PIP install packages from "requirements.txt"
- Download zipped dataset (source: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data)

## Program Description (Doc String)
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
