import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RiskPredictor:
    def __init__(self, model_path='random_forest.pkl'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def load_or_train_mock_model(self):
        '''
        Loads existing model if present, otherwise trains a mock model since we don't
        have physical ground truth data yet.
        '''
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True
        else:
            print("Training mock ML model...")
            self._train_mock()

    def _train_mock(self):
        '''
        Trains the model on synthesized feature vectors.
        '''
        # 15 features (3 aggregations * 5 spatial features)
        X_mock = np.random.rand(100, 15)
        # Random binary targets: 0 = Safe, 1 = Risk of Failure
        y_mock = np.random.randint(0, 2, 100)
        
        X_scaled = self.scaler.fit_transform(X_mock)
        self.model.fit(X_scaled, y_mock)
        self.is_trained = True
        
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
        print("Mock ML model trained and saved.")

    def predict_risk(self, feature_vector):
        '''
        Outputs a risk score between 0 and 1.
        '''
        if not self.is_trained:
            self.load_or_train_mock_model()
            
        # Ensure correct shape
        features = feature_vector.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Predict probability of class 1 (Risk)
        risk_probability = self.model.predict_proba(features_scaled)[0][1]
        return risk_probability
