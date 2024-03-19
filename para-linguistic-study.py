import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


csv_directory = "./annotations/"

audio_directory = "./Prima/SC_audio_"

languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 

data = []

for language in languages:
    for split in ["train", "test"]:
        csv_filename = f"{language}_{split}.csv"
        csv_path = os.path.join(csv_directory, csv_filename)
        csv_data = pd.read_csv(csv_path)
        
        for index, row in csv_data.iterrows():
            t = audio_directory+language+'/'
            audio_path = os.path.join(t, row['filename'])
            data.append({
                'path_to_audio': audio_path,
                'language': language,
                'train_test': split,
                'abuse': row['label']
            })


df = pd.DataFrame(data)
audio = torch.load('./features/audio/audio_features.pth', map_location=torch.device('cpu')).squeeze(dim=1)
    
audio = pd.DataFrame(audio, columns=[f'audio_feature_{i}' for i in range(32)])
audio_df = pd.concat([audio, df['language'], df['abuse']], axis=1)
audio_df['abuse'] = np.where(audio_df['abuse'].values == 'Yes', 1, 0)


# Abuse Statistics

languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 
for lang in languages:
    print('Distribution of Non-Abuses vs Abuses in ', lang)
    print(pd.DataFrame(df[df['language']==lang].drop(['path_to_audio', 'language'],axis=1).groupby('abuse').size()).T)


def compute_distances(df, languages):
    language_map = {}
    languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"]
    i = 0
    for lang in languages:
        language_map[lang] = i
        i+=1

    df['abuse'] = np.where(df['abuse'].values == 'Yes', 1, 0)
    df['language'] = df['language'].replace(language_map)
    features = df.iloc[:, :193]
    
    # Language Basis

    languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"]
    language_basis = {}
    for language in df['language'].unique():
        language_data = df[df['language'] == language].drop(['language', 'abuse'], axis=1)
        language_basis[languages[language]] = language_data.mean(axis=0)

    min_distance = float('inf')
    min_distance_languages = ('', '')
    eucledian = pd.DataFrame(index=languages, columns=languages)

    # Eucledian Distance

    for lang1 in language_basis:
        for lang2 in language_basis:
            if lang1 != lang2:
                distance = euclidean_distances([language_basis[lang1]], [language_basis[lang2]])[0, 0]
                eucledian.loc[lang1, lang2] = distance

                if distance < min_distance:
                    min_distance = distance
                    min_distance_languages = (lang1, lang2)

    print(f"\nMinimum Eucledian distance is between {min_distance_languages[0]} and {min_distance_languages[1]}: {min_distance}")

    # Cosine Similarity

    languages = list(language_basis.keys())
    similarity_matrix = pd.DataFrame(index=languages, columns=languages)

    for lang1 in languages:
        for lang2 in languages:
            if lang1 != lang2:
                similarity = cosine_similarity([language_basis[lang1]], [language_basis[lang2]])[0, 0]
                similarity_matrix.loc[lang1, lang2] = similarity

    # Cosine Similar Pairs
    printed_pairs = set()  # To keep track of printed pairs


    for lang1 in languages:
        for lang2 in languages:
            if lang1 != lang2:
                similarity = similarity_matrix.loc[lang1, lang2]
                printed_pairs.add((lang1, lang2, similarity))

    most_similar_pairs = list(printed_pairs)
    most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
    most_similar_pairs = most_similar_pairs[::2]

    top_n = 5  # Adjust the value of N as needed
    print(f"\nTop {top_n} most similar language pairs:")
    for pair in most_similar_pairs[:top_n]:
        print(f"{pair[0]} and {pair[1]} with similarity score: {pair[2]}")
        
    return similarity_matrix, eucledian


similarity_matrix, eucledian  = compute_distances(audio_df, languages)

print(pd.DataFrame(similarity_matrix))