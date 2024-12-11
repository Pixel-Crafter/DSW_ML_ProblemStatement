import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class LoanDefaultModel:
    def __init__(self, model_type="RandomForest"):
        self.model_type = model_type
        self.model = None

    def load(self, file_path):
        self.data = pd.read_excel(file_path)

    def preprocess(self):
        self.data.fillna(0, inplace=True)
        self.data = pd.get_dummies(self.data, drop_first=True)
        self.X = self.data.drop("loan_status", axis=1)
        self.y = self.data["loan_status"]

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        if self.model_type == "RandomForest":
            self.model_type = RandomForestClassifier()
        elif self.model_type == "LogisticRegression":
            self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test


    def test(self):
        y_pred = self.model.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))
    
    def predict(self, new_data):
        return self.model.predict(new_data)