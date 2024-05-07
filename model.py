import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from tabulate import tabulate

class colors():
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'

def print_colored_text(text, color):
    return f"{color}{text}{colors.END}"

class Predict():
    def __init__(self, Training_data: str, Target_column: str, Predict_data: str = None) -> pd.DataFrame:
        Training_data = pd.read_csv(Training_data)
        if Predict_data:
            Predict_data = pd.read_csv(Predict_data)
            self.predict_data = Predict_data
        self.target_column_name = Target_column
        self.X = Training_data.drop(columns=[Target_column])
        self.y = Training_data[Target_column]

    def model_training(self, model_name: str = None, learn: pd.DataFrame = None, target: pd.DataFrame = None ) -> None:
        if learn is None:
            learn = self.X
        if target is None:
            target = self.y
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
        
        self.model.fit(learn, target)

    def prediction(self, result_file_name: str, model_name: str = None ):
        predict_data = self.predict_data
        self.model_training(model_name)
        predictions = self.model.predict(predict_data)
        target_column_name = self.target_column_name
        self.predict_data[target_column_name] = predictions
        self.predict_data.to_csv(result_file_name, index=False)
        print(f" File Written {result_file_name}")
    
    def model_evaluation(self,  model_name: str = None):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        alogirthm_name = ["Machine Learning Model:", print_colored_text(model_name, colors.MAGENTA)]

        self.model_training(model_name, X_train, y_train)
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        recall = recall_score(y_test, predictions)

        if accuracy < 0.70:
            output = ["Accuracy of the prediciting training Data", f"{print_colored_text(accuracy, colors.RED)}"], ["Confusion Matrix for the Model", f"{confusion}"], ["Precision Score for the Model", f"{precision}"], ["Recall Score for the Model", f"{recall}"]
        elif 0.70 <= accuracy  <= 0.85:
            output = ["Accuracy of the prediciting training Data", f"{print_colored_text(accuracy, colors.YELLOW)}"], ["Confusion Matrix for the Algorithm", f"{confusion}"], ["Precision Score for the Model", f"{precision}"], ["Recall Score for the Model", f"{recall}"]
        else:
            output = ["Accuracy of the prediciting training Data", f"{print_colored_text(accuracy, colors.GREEN)}"], ["Confusion Matrix for the Algorithm", f"{confusion}"], ["Precision Score for the Model", f"{precision}"], ["Recall Score for the Model", f"{recall}"]

        print(tabulate(output, headers=alogirthm_name, tablefmt="grid"))


# result = Predict('TrainingDataBinary.csv', 'marker', 'TestingDataBinary.csv')
# result.prediction('TrainingResultBinary.csv', 'RandomForestClassifier')

evaluate = Predict('TrainingDataBinary.csv', 'marker')
evaluate.model_evaluation('RandomForestClassifier')
