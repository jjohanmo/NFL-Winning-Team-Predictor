# NFL-Winning-Team-Predictor

NFL Winning Team Predictor

Introduction:

NFL scores dataset for “home_team” and “away_team” for games played from 1996 to 2021 was used to develop a full stack application that lets the user predict which NFL team is most likely to win. 

The following are the main objectives of this project:

•	Using NFL team scores dataset to predict winning team
•	Train an ANN model to make accurate predictions
•	Designing an User Interface that users can use to predict the winning team.


Architecture of the application:

 ![image](https://user-images.githubusercontent.com/77942151/208197611-12c95a9a-8abe-4658-aeed-f43d1577f4ac.png)


Data Exploration:

•	Initial shape of dataset: 13232, 17
•	Datatypes of feature: object (8), integer (1), float (6) and boolean (2)
•	Total number of teams: 41
•	Number of stadiums: 113
•	Schedule season: 1966 to 2021
•	Null value analysis:

Column	Number of null values
team_favourite_id	2479
spread_favourite_id	2479
over_under_line	2489
weather_temperature	1043
weather_wind_mph	1060
weather_humidity	4791
weather_detail	10410


Data preprocessing:

•	Removed rows with no scores for “home_team” and “away_team”
•	Handled duplicate team names
•	Created a target variable called ‘label’ that indicates winning team for every game

 

![image](https://user-images.githubusercontent.com/77942151/208197642-8eb9533d-deeb-4bc1-bdd1-bdfe3628aff3.png)


•	Features selected for building the model: 
	Schedule_week, 
	Team_home,
	Team_away
	Playoff
•	Target feature: Label
•	If label is 1 home team wins
•	If label is 0 home team losses


Model Selection

•	ANN model is selected for the project
•	Two layers: One with activation function - “relu” and other with activation function- “sigmoid”
•	Optimizer: Adam
•	Loss: binary- crossentropy
•	Metrics: accuracy 
•	We are storing the trained model checkpoints, so that we can use the best weight for testing the model.

Model Evaluation Metrics and Results:

•	After multiple experiments, the behavior of the model was analyzed under multiple architectures
•	The comparison between training, validation and testing was taken into account for making a decision
•	In the end, the most balanced model was selected for deployment from the backend

The following was the best model out of all the experiments conducted:

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 100)               8800      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101       
================================================================
Total params: 8,901
Trainable params: 8,901
Non-trainable params: 0

Model Results:

•	Epoch 00049: accuracy improved from 0.78673 to 0.79405, saving model to checkpoints\weights-checkpoint-49-0.7941.hdf5
•	Epoch 50/50
•	214/214 - 0s - loss: 0.4408 - accuracy: 0.7886
•	[VALIDATION ]val_loss: 0.8734 - val_accuracy: 0.5354
•	[TESTING] loss: 0.8744 - accuracy: 0.5380

 

User Interface:
 ![image](https://user-images.githubusercontent.com/77942151/208197662-89d63b90-63e5-4d7d-b647-facb6636afdc.png)


Commands to Run the application:

Frontend:
npm install
npm start

Backend:
export FLASK_APP=app.py
flask run





