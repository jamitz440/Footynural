import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from io import StringIO

with open('cleaned_data.csv', 'r', encoding="utf8") as csvfile:
    data = csvfile.read()


df = pd.read_csv(StringIO(data), header=None)
df.columns = ['team1', 'score1', 'score2', 'team2', 'date']

df['outcome'] = np.where((df['score1'] + df['score2']) > 2.5, 0, 1)


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

model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

y_pred = np.round(model.predict(X_test)).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy on test set: {:.2f}%".format(accuracy * 100))

conf_mat = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_mat)

team_dict = {i: team for team, i in encode_teams(df)[['team1', 'team1_encoded']].drop_duplicates().values}

for i in range(len(y_test)):
    team1_encoded = X_test.iloc[i]['team1_encoded']
    team2_encoded = X_test.iloc[i]['team2_encoded']
    try:
        team1 = team_dict[team1_encoded]
    except KeyError:
        team1 = 'Unknown team'
    try:
        team2 = team_dict[team2_encoded]
    except KeyError:
        team2 = 'Unknown team'
    actual = y_test.iloc[i]
    predicted = y_pred[i][0]
    print("Matchup: {} vs. {}. Actual Outcome: {}, Predicted Outcome: {}".format(team1, team2, actual, predicted))

total_accuracy = accuracy_score(y, np.round(model.predict(X)).astype(int))
print("Accuracy on full dataset: {:.2f}%".format(total_accuracy * 100))
