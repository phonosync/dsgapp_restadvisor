import streamlit as st
import io
import pandas as pd
import csv
from surprise import Dataset, Reader, accuracy, SVD, KNNBasic
from surprise.dump import dump, load

import surprise_utils as s_utils

st.title("Restaurants Für Dich")
st.write("""Ein Empfehlungsdienst für Restaurants basierend auf Bewertungen
         von Studierenden der ZHAW School of Engineering""")
fp_ratings = 'ratings.csv'
fp_restaurants = 'restaurants.csv'

# load ratings

df_ratings = pd.read_csv(fp_ratings, delimiter=';')

with open(fp_restaurants) as f:
    dialect = csv.Sniffer().sniff(f.read())
df_rests = pd.read_csv(fp_restaurants, dialect=dialect, )

df_rests = df_rests[df_rests['OSM_ID'].isin(df_ratings['OSM_ID'].unique())]

df_rests['RATING'] = [0]*len(df_rests)

print(df_rests.columns)

# load trained surprise alogorithm

# provide new ratings
st.write('Bewerten Sie drei Restaurants in der Spalte RATING')

edited_df = st.data_editor(df_rests[['RATING', 'name', 'cuisine', 'addr:street', 'addr:housenumber', 'OSM_ID']],
                           key='bla') # 👈 An editable dataframe

if edited_df['RATING'][edited_df['RATING'] > 0].count() > 2:
    

    df_new_user = edited_df.nlargest(3, 'RATING')[['OSM_ID','RATING']] # edited_df.loc[edited_df["RATING"].idxmax()]["OSM_ID"]


    df_new_user['USER_ID'] = ['new_user', 'new_user', 'new_user']
    # st.markdown(f"Your favorite restaurants are **{favorite_command}** 🎈")

    df_new_user = df_new_user[['USER_ID', 'OSM_ID', 'RATING']]
    

    # df_total = df.append(df_new_user, ignore_index=True)
    df_tmp = pd.concat([df_ratings, df_new_user], ignore_index = True)
    # df_total.reset_index()

    # df_tmp

    #retrain
    #algo_tmp = load('algo.pickle')[1]

    algo_tmp = SVD()

    reader = Reader(rating_scale=(1, 5))
    data_tmp = Dataset.load_from_df(df_tmp[["USER_ID", "OSM_ID", "RATING"]], reader)
    trainset_tmp = data_tmp.build_full_trainset()
    algo_tmp.fit(trainset_tmp)

    # Then predict ratings for all pairs (u, i) that are NOT in the training set.
    anti_testset_tmp = trainset_tmp.build_anti_testset()
    predictions_tmp = algo_tmp.test(anti_testset_tmp)

    top_n_tmp = s_utils.get_top_n(predictions_tmp, n=3)

    st.write('Empfehlungen für Sie:')
    bla = df_rests[df_rests['OSM_ID'].isin([rec[0] for rec in top_n_tmp['new_user']])]

    st.dataframe(bla[['RATING', 'name', 'cuisine', 'addr:street', 'addr:housenumber']])

    st.map(bla, size=10)