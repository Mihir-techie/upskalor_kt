
import pandas as pd

data=pd.read_csv("fitlife_emotional_dataset.csv")

data

data.head()

data.tail()

data.shape

data.info()

data['Primary Emotion'].unique()

data.isnull().sum()

data.dtypes

for i in data.columns:
    print(i,':','\n',data[i].unique())

import warnings
warnings.filterwarnings('ignore')

ip=data.drop(['Age','Gender','Time of Day','Activity Category','Activity','Duration (minutes)','Intensity','Secondary Emotion','Mood Before (1-10)'],axis=1)

for value in ip.columns:
    print(value,":", sum(ip[value] == '?'))

ip

data1=ip.rename(columns={'Energy Level (1-10)': 'energy_level',
                         'Mood After (1-10)': 'mind_clarity',
        'Stress Level (1-10)': 'stress_level',
        'Primary Emotion': 'emotion_state'})

data1

from sklearn.preprocessing import LabelEncoder

le1=LabelEncoder()
data1['emotion_state']=le1.fit_transform(data1['emotion_state'])

data1

id = data1['ID'].value_counts().index[0]

id

historical_df = data1[data1['ID'] == id].copy()

historical_df

historical_df['Date'] = pd.to_datetime(historical_df['Date'])
historical_df

historical_df = historical_df.sort_values('Date').set_index('Date')

historical_df

from sklearn.preprocessing import MinMaxScaler


def calculate_user_baseline(file_path):
    
    
    df = pd.read_csv(file_path)


    df = df.drop([
        'Age', 'Gender', 'Time of Day', 'Activity Category', 'Sub-Category',
        'Activity', 'Duration (minutes)', 'Intensity', 'Secondary Emotion',
        'Mood Before (1-10)'
    ], axis=1)


    df = df.rename(columns={
        'Mood After (1-10)': 'mind_clarity',
        'Energy Level (1-10)': 'energy_level',
        'Stress Level (1-10)': 'stress_level',
        'Primary Emotion': 'emotion_state'
    })


    top_id = df['ID'].value_counts().index[0]
    historical_df = df[df['ID'] == top_id].copy()


    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    historical_df = historical_df.sort_values('Date').set_index('Date')


    emotion_map = {
        'Distressed': 1, 'Exhausted': 1, 'Anxious': 1, 'Stressed': 2,
        'Tired': 2, 'Bored': 2, 'Calm': 3, 'Content': 3, 'Relaxed': 4,
        'Serene': 4, 'Happy': 5, 'Energized': 5, 'Accomplished': 5,
        'Invigorated': 5, 'Strong': 5, 'Empowered': 5, 'Fulfilled': 5,
        'Flexible': 4, 'Challenged': 3, 'Refreshed': 4, 'Understood': 4,
        'Rejuvenated': 5, 'Recharged': 4, 'Agile': 4

    }
    historical_df['emotion_score'] = historical_df['emotion_state'].map(emotion_map)


    historical_df = historical_df.drop(columns=['ID', 'emotion_state'])


    features = ['energy_level', 'mind_clarity', 'stress_level', 'emotion_score']
    scaler = MinMaxScaler()

    historical_df[features] = scaler.fit_transform(historical_df[features])


    baseline_mean = historical_df[features].mean().to_frame(name='Baseline_Mean')
    baseline_variance = historical_df[features].var().to_frame(name='Baseline_Variance')


    baseline_stats = pd.concat([baseline_mean, baseline_variance], axis=1)

    return baseline_stats


baseline_results = calculate_user_baseline("fitlife_emotional_dataset.csv")


baseline_results.to_csv('user_baseline_stats.csv')

print(" Personal Baseline Statistics ")
print(baseline_results)
