import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df_cleaned = df.drop(columns=['Unnamed: 0'])
    df_cleaned[['Rating', 'Review Counts']] = df_cleaned['Review counts'].str.extract(r'(\d+\.\d+)\((\d+[KM]?)\sreviews\)')

    def convert_review_counts(value):
        if pd.isna(value):
            return 0
        if 'K' in value:
            return int(float(value.replace('K', '')) * 1000)
        elif 'M' in value:
            return int(float(value.replace('M', '')) * 1000000)
        else:
            try:
                return int(value)
            except ValueError:
                return 0

    df_cleaned['Review Counts'] = df_cleaned['Review Counts'].apply(convert_review_counts)
    df_cleaned['Rating'] = df_cleaned['Rating'].astype(float)
    df_cleaned = df_cleaned.drop(columns=['Review counts'])
    return df_cleaned

# Load and preprocess data
file_path = 'coursera_course_dataset_v2_no_null.csv'
df_cleaned = preprocess_data(file_path)

# Save cleaned dataframe
with open('df_cleaned.pkl', 'wb') as file:
    pickle.dump(df_cleaned, file)

# Combine features for the TF-IDF vectorizer
df_cleaned['Combined Features'] = df_cleaned['Title'] + ' ' + df_cleaned['Organization'] + ' ' + df_cleaned['Skills']

# Create the TF-IDF vectorizer and compute the cosine similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cleaned['Combined Features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Save TF-IDF matrix and cosine similarity matrix
with open('tfidf_matrix.pkl', 'wb') as file:
    pickle.dump(tfidf_matrix, file)

with open('cosine_sim.pkl', 'wb') as file:
    pickle.dump(cosine_sim, file)

# Create a pivot table of user ratings and decompose the matrix using SVD
pivot_table = df_cleaned.pivot_table(index='Title', columns='Rating', values='Review Counts', fill_value=0)
svd = TruncatedSVD(n_components=12, random_state=42)
matrix_svd = svd.fit_transform(pivot_table)
cosine_sim_svd = linear_kernel(matrix_svd, matrix_svd)

# Save pivot table and SVD cosine similarity matrix
with open('pivot_table.pkl', 'wb') as file:
    pickle.dump(pivot_table, file)

with open('cosine_sim_svd.pkl', 'wb') as file:
    pickle.dump(cosine_sim_svd, file)
