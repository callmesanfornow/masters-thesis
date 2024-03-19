import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.gaussian_process.kernels import RBF

# Path to the directory containing CSV files
csv_directory = "./annotations/"

# Path to the directory containing audio directories
audio_directory = "./Prima/SC_audio_"

languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 

# Initialize an empty list to store data
data = []

# Iterate through each language and train/test combination
for language in languages:
    for split in ["train", "test"]:
        # Read the CSV file
        csv_filename = f"{language}_{split}.csv"
        csv_path = os.path.join(csv_directory, csv_filename)
        csv_data = pd.read_csv(csv_path)
        
        # Iterate through each row in the CSV data
        for index, row in csv_data.iterrows():
            t = audio_directory+language+'/'
            audio_path = os.path.join(t, row['filename'])
            data.append({
                'path_to_audio': audio_path,
                'language': language,
                'train_test': split,
                'abuse': row['label']
            })

# Create a DataFrame from the collected data
df = pd.DataFrame(data)
audio = torch.load('./features/audio/audio_features.pth', map_location=torch.device('cpu')).squeeze(dim=1)
    
emo = torch.load('./features/emotion/emotion_features.pth', map_location=torch.device('cpu')).squeeze(dim=1)

text = []
for language in languages:
    temp = torch.load(f'./features/text/{language}-text-emb.pth', map_location=torch.device('cpu'))
    text.append(temp)
text = torch.vstack(text)


audio_df = pd.DataFrame(audio, columns=[f'audio_feature_{i}' for i in range(32)])
emo_df = pd.DataFrame(emo, columns=[f'emo_feature_{i}' for i in range(193)])
text_df = pd.DataFrame(text, columns=[f'text_feature_{i}' for i in range(768)])
merged_df = pd.concat([df['train_test'],df['language'],audio_df, emo_df, text_df, df['abuse']], axis=1)

def PCA_features(df):
    # Fit the scaler to your data and transform it
    scaled_data = scaler.fit_transform(df.values)
    pca = PCA()
    pca.fit(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print('Number of PCA Components contributing to 95% variance :', num_components)
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(scaled_data)
    
    return principal_components, num_components

pca_full, num_components = PCA_features(pd.concat([audio_df, emo_df, text_df], axis=1))
columns = [f'PC{i+1}' for i in range(num_components)]
full = pd.DataFrame(data=pca_full, columns=columns)

pca_df = pd.concat([full, merged_df.abuse, merged_df.language, merged_df.train_test], axis=1)
pca_df.abuse = np.where(pca_df.abuse.values == 'Yes', 1, 0)

train = pca_df[pca_df['train_test']=='train'].drop(['train_test'], axis=1)
test = pca_df[pca_df['train_test']=='test'].drop(['train_test'], axis=1)


train_dict = {}
test_dict = {}

languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 
for language in languages:
    train_temp_dict = {}
    train_temp_dict['feat'] = train[train['language']==language].drop(['abuse', 'language'], axis=1)
    train_temp_dict['label'] = train[train['language']==language]['abuse']
    train_dict[language] = train_temp_dict
    test_temp_dict = {}
    test_temp_dict['feat'] = test[test['language']==language].drop(['abuse', 'language'], axis=1)
    test_temp_dict['label'] = test[test['language']==language]['abuse']
    test_dict[language] = test_temp_dict

X_train = []
X_test = []
y_train = []
y_test = []
for language in languages:
    X_train.append(train_dict[language]['feat'])
    y_train.append(train_dict[language]['label'])
    X_test.append(test_dict[language]['feat'])
    y_test.append(test_dict[language]['label'])


# Mono-Lingual Tests

for language in enumerate(languages):

    print(f"Language =  {language[0]}")

    # ADIMA Classifier (AC)
    ac_classifier = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', alpha=0.0001,
                              learning_rate_init=0.001, max_iter=50, random_state=42)
    ac_classifier.fit(X_train[language[0]], y_train[language[0]])
    

    print('-'*63)
    predictions = ac_classifier.predict(X_test[language[0]])
    report = classification_report(y_test[language[0]], predictions, target_names=['No', 'Yes'], output_dict=True)
    print(f"ADIMA-Classifier Classification Report for {language[0]} Mono-Lingual abuse detection:\n", 'Accuracy: {:.2f}'.format(report['accuracy']), '\nMacro F1: {:.2f}'.format(report['macro avg']['recall']))

    # Stack Classifier
    base_classifiers = [
        ('gp', GaussianProcessClassifier(kernel=1.0 * RBF(1.0))),
        ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)),
        ('svm', SVC(kernel='linear', C=0.025)),
        ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
        ('lr', LogisticRegression())
    ]

    sc_classifier = StackingClassifier(estimators=base_classifiers,
                                       final_estimator=LogisticRegression(),
                                       cv=5)
    
    sc_classifier.fit(X_train[language[0]], y_train[language[0]])
    print('-'*63)
    predictions = sc_classifier.predict(X_test[language[0]])
    report = classification_report(y_test[language[0]], predictions, target_names=['No', 'Yes'], output_dict=True)
    print(f"Stack-Classifier Classification Report for {language[0]} Mono-Lingual abuse detection:\n", 'Accuracy: {:.2f}'.format(report['accuracy']), '\nMacro F1: {:.2f}'.format(report['macro avg']['recall']))


# Cross-Lingual Tests
    
X_train = train.drop(['abuse', 'language'], axis=1)
y_train = train['abuse']
X_test = {}
y_test = {}
languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 

for lang in languages:
    X_test[lang] = test[test['language']==lang].drop(['abuse', 'language'], axis=1)
    y_test[lang] = test[test['language']==lang]['abuse']

# Stack Classifier (SC)
base_classifiers = [
    ('gp', GaussianProcessClassifier(kernel=1.0 * RBF(1.0))),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)),
    ('svm', SVC(kernel='linear', C=0.025)),
    ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
    ('lr', LogisticRegression())
]

sc_classifier = StackingClassifier(estimators=base_classifiers,
                                   final_estimator=LogisticRegression(),
                                   cv=5)
sc_classifier.fit(X_train, y_train)

languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 

for lang in languages:
    print('-'*63)
    predictions = sc_classifier.predict(X_test[lang])
    report = classification_report(y_test[lang], predictions, target_names=['No', 'Yes'], output_dict=True)
    print(f"Stack-Classifier Classification Report for {lang} Cross-Lingual abuse detection:\n", 'Accuracy: {:.2f}'.format(report['accuracy']), '\nMacro F1: {:.2f}'.format(report['macro avg']['recall']))

ac_classifier = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', alpha=0.0001,
                              learning_rate_init=0.001, max_iter=50, random_state=42)
ac_classifier.fit(X_train, y_train)

for lang in languages:
    print('-'*63)
    predictions = ac_classifier.predict(X_test[lang])
    report = classification_report(y_test[lang], predictions, target_names=['No', 'Yes'], output_dict=True)
    print(f"ADIMA Classification Report for {lang} Cross-Lingual abuse detection:\n", 'Accuracy: {:.2f}'.format(report['accuracy']), '\nMacro F1: {:.2f}'.format(report['macro avg']['recall']))

# Classification Task Complete ✅