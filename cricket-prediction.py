import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Python program showing 
# a use of raw_input() 

team1 = input("Enter name of team1 : ") 
team2 = input("Enter name of team2 : ") 



df = pd.read_csv('file.csv')
df = df[df["Margin"].str.contains("wickets")==True]
df["Margin"] = df["Margin"].str.replace(" wickets","")
df[['team1','team2']] = df.Match.str.split(" v ",expand=True) 

df["won"] = np.where((df['Result'] == 'Won') , df["Country"],np.where((df['Result'] == 'Lost') & (df["team1"]==df["Country"]), df["team2"], df["team1"]))


cleanup_nums = {"team1":   
{"India": 0, "Australia": 1, "New Zealand": 2
 , "England": 3, "Bangladesh": 4, "South Africa": 5, 
 "Sri Lanka": 6, "West Indies": 7, "Zimbabwe": 8, "Pakistan": 9
 , "Ireland": 10, "Afghanistan": 11, "Kenya": 12,
 "Scotland": 13, "Netherlands": 14, "Canada": 15,
 "Bermuda": 16, "U.A.E.": 17,
 "P.N.G.": 18, "Hong Kong": 19,
 "Namibia": 20, "U.S.A.": 21,"Oman":22},
 "team2": {"India": 0, "Australia": 1, "New Zealand": 2
 , "England": 3, "Bangladesh": 4, "South Africa": 5, 
 "Sri Lanka": 6, "West Indies": 7, "Zimbabwe": 8, "Pakistan": 9
 , "Ireland": 10, "Afghanistan": 11, "Kenya": 12,
 "Scotland": 13, "Netherlands": 14, "Canada": 15,
 "Bermuda": 16, "U.A.E.": 17,
 "P.N.G.": 18, "Hong Kong": 19,
 "Namibia": 20, "U.S.A.": 21 ,"Oman":22},
   "won":{"India": 0, "Australia": 1, "New Zealand": 2
 , "England": 3, "Bangladesh": 4, "South Africa": 5, 
 "Sri Lanka": 6, "West Indies": 7, "Zimbabwe": 8, "Pakistan": 9
 , "Ireland": 10, "Afghanistan": 11, "Kenya": 12,
 "Scotland": 13, "Netherlands": 14, "Canada": 15,
 "Bermuda": 16, "U.A.E.": 17,
 "P.N.G.": 18, "Hong Kong": 19,
 "Namibia": 20, "U.S.A.": 21,"Oman":22}}
 
#print(list(cleanup_nums["team1"].keys()))
 
df.replace(cleanup_nums, inplace=True)
X_train = df[['team1',"team2"]]
y_train = df[['won']]

X_train,X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.3, random_state = 120)

model = RandomForestClassifier(n_estimators=50, 
                               bootstrap = True,
                               max_features = 'sqrt')
model.fit(X_train,y_train)
print()
#select team
team1 = cleanup_nums["team1"][""+str(team1)]
team2 = cleanup_nums["team1"][""+str(team2)]
team_names = list(cleanup_nums["team1"].keys())
print("Team 1:",team_names[team1])
print("Team 2:",team_names[team2])
predictions = model.predict(np.array([[team1,team2]]))
print("Team won: ",team_names[predictions[0]])

X_test = model.predict(X_test)

percentage = model.predict_proba(np.array([[team1,team2]]))
print("Winning percentage of ",team_names[team1],":",percentage[0][team1])
print("Winning percentage of ",team_names[team2],":",percentage[0][team2])
filename = 'cricket_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Accuracy Random Forest on unseen data: ",accuracy_score(y_test, X_test)*100)