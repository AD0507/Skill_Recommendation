import streamlit as st
import pandas as pd
import pickle

# Load preprocessed data and models
with open('df_cleaned.pkl', 'rb') as file:
    df_cleaned = pickle.load(file)

with open('cosine_sim.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('pivot_table.pkl', 'rb') as file:
    pivot_table = pickle.load(file)

with open('cosine_sim_svd.pkl', 'rb') as file:
    cosine_sim_svd = pickle.load(file)

# Content-based recommendation function
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    indices = pd.Series(df_cleaned.index, index=df_cleaned['Title']).drop_duplicates()
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    course_indices = [i[0] for i in sim_scores]
    return df_cleaned.iloc[course_indices][['Title', 'Organization', 'Rating', 'Review Counts']]

# Collaborative filtering recommendation function
def get_collaborative_recommendations(title, cosine_sim=cosine_sim_svd):
    indices = pd.Series(pivot_table.index)
    idx = indices[indices == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    course_indices = [i[0] for i in sim_scores]
    return df_cleaned.iloc[course_indices][['Title', 'Organization', 'Rating', 'Review Counts']]

# Hybrid recommendation function
def hybrid_recommendations(title):
    content_recommend = get_content_based_recommendations(title)
    collaborative_recommend = get_collaborative_recommendations(title)
    combined_recommend = pd.concat([content_recommend, collaborative_recommend]).drop_duplicates().head(10)
    return combined_recommend

# Streamlit App
st.title('Course Recommendation System')

# Select a course
course_list = df_cleaned['Title'].unique()
selected_course = st.selectbox('Select a course:', course_list)

if st.button('Get Recommendations'):
    # Content-based recommendations
    st.subheader('Content-Based Recommendations')
    content_recommendations = get_content_based_recommendations(selected_course)
    st.write(content_recommendations)
    
    # Collaborative filtering recommendations
    st.subheader('Collaborative Filtering Recommendations')
    collaborative_recommendations = get_collaborative_recommendations(selected_course)
    st.write(collaborative_recommendations)
    
    # Hybrid recommendations
    st.subheader('Hybrid Recommendations')
    hybrid_recommendations_df = hybrid_recommendations(selected_course)
    st.write(hybrid_recommendations_df)
    
    # Top 2 recommendations based on all three approaches
    st.subheader('Top 2 Recommendations')
    combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations, hybrid_recommendations_df])
    top_2_recommendations = combined_recommendations.drop_duplicates().head(2)
    st.write(top_2_recommendations)
