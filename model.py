import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Predict():
    def __init__(self, Training_data: str, Target_column: str, Predict_data: str = None) -> pd.DataFrame:
        Training_data = pd.read_csv(Training_data)
        Predict_data = pd.read_csv(Predict_data)
        self.predict_data = Predict_data
        self.target_column_name = Target_column
        self.X = Training_data.drop(columns=[Target_column])
        self.y = Training_data[Target_column]

    def model_training(self, model_name: str = None) -> None:
        if not model_name:
            print('Model Name not provided')
            print('Model Names available:\n\t==> RandomForestClassifier')
            exit(1)
        # if model_name == 'DecisionTreeClassifier':
        #     model = DecisionTreeClassifier()
        if model_name == 'RandomForestClassifier':
            self.model = RandomForestClassifier()
        else:
            print('UMM UMM Check the model name again')
            exit(1)
        
        self.model.fit(self.X, self.y)

    def prediction(self, result_file_name: str, model_name: str = None ):
        predict_data = self.predict_data
        self.model_training(model_name)
        predictions = self.model.predict(predict_data)
        target_column_name = self.target_column_name
        self.predict_data[target_column_name] = predictions
        self.predict_data.to_csv(result_file_name, index=False)
        print(f" File Written {result_file_name}")


result = Predict('TrainingDataBinary.csv', 'marker', 'TestingDataBinary.csv')
result.prediction('TrainingResultBinary.csv', 'RandomForestClassifier')