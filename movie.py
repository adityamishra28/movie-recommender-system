import pandas as pd
import numpy as np
df1 = pd.read_csv("C:\\Users\\Aditya\\Documents\\tmdb_5000_credits\\credits.csv")
df2 = pd.read_csv("C:\\Users\\Aditya\\Documents\\tmdb_5000_movies\\movies.csv")
df1.column = ['id','tittle','cast','crew']
df2 = df2.merge(df1,on='id')
df2.head(5)
c = df2['vote_average'].mean()
c
m = df2['vote_count'].quantile(0.9)
m  
q_movies = df2.copy().loc[df2['vote count'] >= m]
q_movies.shape
def weighted_rating(x, m=m, c=c):
    v = x['vote_count']
    r = x['vote_average']
    return (v/(v+m) * r) + (m/(m+v) * c)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending = False)
q_movies[['title','vote count','vote average','score']].head(10)
pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
#Content Based Filtering
df2['overview'].head(5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]
get_recommendations('The Dark Knight Rises')
#Credits, Genres and Keywords Based Recommender
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []
df2['director'] = df2['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)
rom sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
get_recommendations('The Dark Knight Rises', cosine_sim2)
#Collaborative Filtering
from surprise import Reader, Dataset, SVD, evaluate
reader = Reader()
ratings = pd.read_csv("C:\\Users\\Aditya\\Documents\\tmdb_5000_ratings\\ratings_small.csv")
ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)