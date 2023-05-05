from bs4 import BeautifulSoup
import requests
import datetime
import pickle

data = []

def milkDate(url):
    global data
    URL = url
    page = requests.get(URL)
    page = page.content
    soup = BeautifulSoup(page, 'html.parser')

    results = soup.find_all('ul', class_='CompetitionList__CompetitionListWrapper-sc-1f2woz6-0 PJXXy')
    teamA = soup.find_all('span', class_='Item__TeamA-et8305-6 bZRnjo')
    teamScore = soup.find_all('span', class_='Item__TeamsModifier-et8305-7 fuOChu')
    teamB = soup.find_all('span', class_='Item__TeamB-et8305-8 dqXJJw')

    teamAList = []
    teamAScore = []
    teamBList = []
    teamBScore = []

    for i in teamA:
        teamAList.append(i.text)
    for i in teamScore:
        teamAScore.append(i.text[:1])
        teamBScore.append(i.text[-1])
    for i in teamB:
        teamBList.append(i.text)

    date = URL[-10:]

    data = [(a, b, c, d, date) for a, b, c, d in zip(teamAList, teamAScore, teamBScore, teamBList)]

    with open('data.pkl', 'rb') as file:
        existing_data = pickle.load(file)

        # Add new_data to the existing_data
        existing_data += data

        # Write updated data to a file
        with open('data.pkl', 'wb') as file:
            pickle.dump(existing_data, file)

def getURLs():
    url = "https://www.sportinglife.com/football/fixtures-results/DATE"
    complete = 0
    URLs = []
    numberOfDaysToMilk = 6200
    datetime_object = datetime.datetime.now()

    for i in range(numberOfDaysToMilk): # 6200
        datetime_object = datetime_object - datetime.timedelta(days=1)
        date = (str(datetime_object).split(" ")[0])
        URLs.append(url.replace("DATE", date))

    for i in URLs:
        milkDate(i)
    
    print ("Done")

getURLs()
