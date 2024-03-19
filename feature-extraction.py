import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm
import torch
import os
import pandas as pd
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline, AutoTokenizer, AutoModel
import librosa
import librosa.feature
import re



# Path to the directory containing CSV files
csv_directory = "./annotations/"

# Path to the directory containing audio directories
audio_directory = "./Prima/SC_audio_"

# List of language names
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

df = pd.DataFrame(data)
# Set Device to CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Audio Features


languages = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 
embeddings = []

models = {
        'Tamil' : 'Harveenchadha/vakyansh-wav2vec2-tamil-tam-250',
        'Hindi' : 'Harveenchadha/hindi_base_wav2vec2',
        'Gujarati' : 'Harveenchadha/vakyansh-wav2vec2-gujarati-gnm-100',
        'Kannada' : 'Harveenchadha/vakyansh-wav2vec2-kannada-knm-560',
        'Bengali' : 'Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200',
        'Punjabi' : 'Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10',
        'Odia' : 'Harveenchadha/vakyansh-wav2vec2-odia-orm-100',
        'Malayalam' : 'Harveenchadha/vakyansh-wav2vec2-malayalam-mlm-8',
    }


def audio_extraction_module(df, lang):
    embeddings = []
    temp = df[df['language']==lang].drop(['language', 'train_test', 'abuse'], axis=1)
    processor = Wav2Vec2ForCTC.from_pretrained(models[lang]).to(device)

    for path in tqdm(temp['path_to_audio']):
        waveform, sample_rate = torchaudio.load(path)
        # Preprocess audio and extract features
        with torch.no_grad():
            input_values = processor(waveform.to(device))
            input_values = input_values.logits.cpu().detach().numpy()   
            embeddings.append(input_values)

    torch.cuda.empty_cache()
    return embeddings

for lang in languages:
    embeddings.append(audio_extraction_module(df, lang))

x=[]
for i in tqdm(range(len(embeddings))):
    x.append(np.mean(embeddings[i],axis=1))

mean_pooled = torch.tensor(x)
torch.save(mean_pooled, './features/audio/audio_features.pth')

torch.cuda.empty_cache()


# Emotion Features

result = []
i=0
for audio in tqdm(df['path_to_audio']):
    
    X, sample_rate = librosa.load(audio)
    stft = np.abs(librosa.stft(X))
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    temp = np.hstack((mfccs, chroma))
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
    temp = np.hstack((temp, mel))
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    temp = np.hstack((temp, contrast))
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    result.append(np.hstack((temp, tonnetz)))

torch.save(torch.tensor(result), 'emotion_features.pth')

# Text Features

## Step 1

def transcription(df, transcribe):
    sentences = []
    i= 0 
    
    for audio in tqdm(df['path_to_audio']):
        sentences.append(transcribe(audio)['text'])
    return sentences
        
data = {
        'Tamil' : ['ta', 'Harveenchadha/vakyansh-wav2vec2-tamil-tam-250'],
        'Hindi' : ['hi', 'Harveenchadha/hindi_base_wav2vec2'],
        'Gujarati' : ['gu', 'Harveenchadha/vakyansh-wav2vec2-gujarati-gnm-100'],
        'Kannada' : ['kn', 'Harveenchadha/vakyansh-wav2vec2-kannada-knm-560'],
        'Bengali' : ['bn', 'Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200'],
        'Punjabi' : ['pa', 'Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10'],
        'Odia' : ['or', 'Harveenchadha/vakyansh-wav2vec2-odia-orm-100'],
        'Malayalam' : ['ma', 'Harveenchadha/vakyansh-wav2vec2-malayalam-mlm-8'],
    }


for language in languages:
    transcribe = pipeline(task="automatic-speech-recognition", model=data[language][1], chunk_length_s=30, device='cuda:0')
    # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=data[language][0], task="transcribe")
    tempdf = df[df['language']==language].drop(['language', 'abuse', 'train_test'], axis=1)
    tempdf = tempdf.head(15)
    sentences = transcription(tempdf, transcribe)
    
    file_path = f"./transcription/{language}_transcription.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        for string in sentences:
            file.write(string + "\n")
        
    torch.cuda.empty_cache()


# Step 2
    
file_path = "./transcription/"
path_list = []
for i in range(len(languages)):
    path_list.append(file_path+languages[i]+'_transcription.txt')

sentences = {}

for i in range(len(languages)):
    with open(path_list[i], "r", encoding="utf-8") as file:
        for line in file:
            # Remove the newline character and add the string to the list
            if languages[i] not in sentences.keys():
                sentences[languages[i]] = [line.strip()]
            else:
                sentences[languages[i]].append(line.strip())


for language in sentences.keys():
    for i in range(len(sentences[language])):
        sentences[language][i] = re.sub(r'<s>', '', sentences[language][i])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('l3cube-pune/indic-sentence-bert-nli')
model = AutoModel.from_pretrained('l3cube-pune/indic-sentence-bert-nli').to('cuda')

for lang in languages:
    encoded_input = tokenizer(sentences[lang], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input.to('cuda'))

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    torch.save(sentence_embeddings, f'{lang}-text-emb.pth')
    torch.cuda.empty_cache()

# Feature Extraction Complete ✅