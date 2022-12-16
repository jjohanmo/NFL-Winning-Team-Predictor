import numpy as np
import pandas as pd
import tensorflow as tf
import os

schedule_week_map = {
    '1': 'schedule_week_1',
    '10': 'schedule_week_10',
    '11': 'schedule_week_11',
    '12': 'schedule_week_12',
    '13': 'schedule_week_13',
    '14': 'schedule_week_14',
    '15': 'schedule_week_15',
    '16': 'schedule_week_16',
    '17': 'schedule_week_17',
    '18': 'schedule_week_18',
    '2': 'schedule_week_2',
    '3': 'schedule_week_3',
    '4': 'schedule_week_4',
    '5': 'schedule_week_5',
    '6': 'schedule_week_6',
    '7': 'schedule_week_7',
    '8': 'schedule_week_8',
    '9': 'schedule_week_9',
    'conference': 'schedule_week_conference',
    'division': 'schedule_week_division',
    'superbowl': 'schedule_week_superbowl',
    'wildcard': 'schedule_week_wildcard'
}

teams_home_map = {
    'Arizona Cardinals': 'team_home_Arizona Cardinals',
    'Atlanta Falcons': 'team_home_Atlanta Falcons',
    'Baltimore Ravens': 'team_home_Baltimore Ravens',
    'Buffalo Bills': 'team_home_Buffalo Bills',
    'Carolina Panthers': 'team_home_Carolina Panthers',
    'Chicago Bears': 'team_home_Chicago Bears',
    'Cincinnati Bengals': 'team_home_Cincinnati Bengals',
    'Cleveland Browns': 'team_home_Cleveland Browns',
    'Dallas Cowboys': 'team_home_Dallas Cowboys',
    'Denver Broncos': 'team_home_Denver Broncos',
    'Detroit Lions': 'team_home_Detroit Lions',
    'Green Bay Packers': 'team_home_Green Bay Packers',
    'Houston Texans': 'team_home_Houston Texans',
    'Indianapolis Colts': 'team_home_Indianapolis Colts',
    'Jacksonville Jaguars': 'team_home_Jacksonville Jaguars',
    'Kansas City Chiefs': 'team_home_Kansas City Chiefs',
    'Los Angeles Chargers': 'team_home_Los Angeles Chargers',
    'Los Angeles Rams': 'team_home_Los Angeles Rams',
    'Miami Dolphins': 'team_home_Miami Dolphins',
    'Minnesota Vikings': 'team_home_Minnesota Vikings',
    'New England Patriots': 'team_home_New England Patriots',
    'New Orleans Saints': 'team_home_New Orleans Saints',
    'New York Giants': 'team_home_New York Giants',
    'New York Jets': 'team_home_New York Jets',
    'Oakland Raiders': 'team_home_Oakland Raiders',
    'Philadelphia Eagles': 'team_home_Philadelphia Eagles',
    'Pittsburgh Steelers': 'team_home_Pittsburgh Steelers',
    'San Francisco 49ers': 'team_home_San Francisco 49ers',
    'Seattle Seahawks': 'team_home_Seattle Seahawks',
    'Tampa Bay Buccaneers': 'team_home_Tampa Bay Buccaneers',
    'Tennessee Titans': 'team_home_Tennessee Titans',
    'Washington Redskins': 'team_home_Washington Redskins',
}

teams_away_map = {
    'Arizona Cardinals': 'team_away_Arizona Cardinals',
    'Atlanta Falcons': 'team_away_Atlanta Falcons',
    'Baltimore Ravens': 'team_away_Baltimore Ravens',
    'Buffalo Bills': 'team_away_Buffalo Bills',
    'Carolina Panthers': 'team_away_Carolina Panthers',
    'Chicago Bears': 'team_away_Chicago Bears',
    'Cincinnati Bengals': 'team_away_Cincinnati Bengals',
    'Cleveland Browns': 'team_away_Cleveland Browns',
    'Dallas Cowboys': 'team_away_Dallas Cowboys',
    'Denver Broncos': 'team_away_Denver Broncos',
    'Detroit Lions': 'team_away_Detroit Lions',
    'Green Bay Packers': 'team_away_Green Bay Packers',
    'Houston Texans': 'team_away_Houston Texans',
    'Indianapolis Colts': 'team_away_Indianapolis Colts',
    'Jacksonville Jaguars': 'team_away_Jacksonville Jaguars',
    'Kansas City Chiefs': 'team_away_Kansas City Chiefs',
    'Los Angeles Chargers': 'team_away_Los Angeles Chargers',
    'Los Angeles Rams': 'team_away_Los Angeles Rams',
    'Miami Dolphins': 'team_away_Miami Dolphins',
    'Minnesota Vikings': 'team_away_Minnesota Vikings',
    'New England Patriots': 'team_away_New England Patriots',
    'New Orleans Saints': 'team_away_New Orleans Saints',
    'New York Giants': 'team_away_New York Giants',
    'New York Jets': 'team_away_New York Jets',
    'Oakland Raiders': 'team_away_Oakland Raiders',
    'Philadelphia Eagles': 'team_away_Philadelphia Eagles',
    'Pittsburgh Steelers': 'team_away_Pittsburgh Steelers',
    'San Francisco 49ers': 'team_away_San Francisco 49ers',
    'Seattle Seahawks': 'team_away_Seattle Seahawks',
    'Tampa Bay Buccaneers': 'team_away_Tampa Bay Buccaneers',
    'Tennessee Titans': 'team_away_Tennessee Titans',
    'Washington Redskins': 'team_away_Washington Redskins',
}

# playoff


class NFLPredictor():

    def __init__(self, model_path, weights_path):
        # self.model = Sequential()
        self.model = tf.keras.models.load_model(model_path)
        self.model.load_weights(weights_path)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

    def __build_input_dataset(self):
        cols = ['playoff']
        cols += list(schedule_week_map.values())
        cols += list(teams_home_map.values())
        cols += list(teams_away_map.values())
        # print(len(cols))
        # print(cols)
        # Create an empty dataframe filled with zeros for initialize the prediction
        self.x = pd.DataFrame(columns=cols)
        self.x = self.x.append(pd.Series(0, index=cols), ignore_index=True)

    def predict(self, home_team_name, away_team_name, is_playoff, schedule_week):
        # Create empty dataframe
        self.__build_input_dataset()
        team_home_col = teams_away_map[home_team_name]
        team_away_col = teams_away_map[away_team_name]
        playoff = 1 if is_playoff else 0
        schedule_col = schedule_week_map[schedule_week]

        # print(team_home_col)
        # print(team_away_col)
        # print(playoff)
        # print(schedule_col)
        self.x[team_home_col] = 1
        self.x[team_away_col] = 1
        self.x['playoff'] = playoff
        self.x[schedule_col] = 1

        # print(self.x)
        # Reshape row to match input expected by model (,87)
        input = self.x.iloc[0].copy()
        # print(input.shape)
        # print(type(input))
        # print(input)
        input = input.values.reshape(1, input.shape[0])
        prediction = self.model.predict(input.astype(np.int32))
        result = 1 if prediction > 0.5 else 0
        return result


# Executable
if __name__ == "__main__":

    # Load model and weights from disk
    model_path = os.path.join(
        os.getcwd(), 'FinalProject', 'nfl_predictor.model')
    weights_path = os.path.join(
        os.getcwd(), 'FinalProject', 'weights_best.hdf5')
    # print(model_path)
    # print(weights_path)

    nfl = NFLPredictor(model_path, weights_path)
    result = nfl.predict(
        'Oakland Raiders', 'Pittsburgh Steelers', 1, 'superbowl')
    print("MODEL RESULT: ", result)