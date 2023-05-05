import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from io import StringIO
import csv

with open('cleaned_data.csv', 'r') as csvfile:
    data = csvfile.read()

df = pd.read_csv(StringIO(data), header=None)
df.columns = ['team1', 'score1', 'score2', 'team2', 'date']

df['outcome'] = np.where(df['score1'] == df['score2'], 0, np.where(df['score1'] > df['score2'], 1, 2))

def encode_teams(df):
    teams = pd.concat([df['team1'], df['team2']]).unique()
    team_dict = {team: i for i, team in enumerate(teams)}
    df['team1_encoded'] = df['team1'].apply(lambda x: team_dict[x])
    df['team2_encoded'] = df['team2'].apply(lambda x: team_dict[x])
    return df

df = encode_teams(df)


X = df[['team1_encoded', 'team2_encoded']]
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
