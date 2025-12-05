import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from typing import Tuple
from config import config

class UpliftModel:
    """
    X-Learner for CATE estimation
    Predicts incremental CSAT from assigning customer to specific agent
    """
    def __init__(self):
        self.mu0_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.mu1_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.tau0_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.tau1_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.is_trained = False
    
    def _featurize(self, customer: pd.Series, agent: pd.Series) -> np.ndarray:
        """Create feature vector for (customer, agent) pair"""
        features = []
        
        # Customer features
        features.append(config.SKILLS.index(customer['skill_needed']))
        features.append(['low', 'medium', 'high', 'vip'].index(customer['priority']))
        features.append(customer['complexity'])
        features.append(customer['wait_time'])
        features.append(config.CHANNELS.index(customer['channel']))
        
        # Agent features
        features.append(agent[f'skill_{customer["skill_needed"]}'])
        features.append(agent['avg_csat'])
        features.append(agent['avg_aht'])
        features.append(agent['experience_years'])
        features.append(agent['current_load'])
        
        # Interaction features
        features.append(1 if customer['channel'] in agent['preferred_channels'] else 0)
        features.append(agent[f'skill_{customer["skill_needed"]}'] * customer['complexity'])
        
        return np.array(features)
    
    def train(self, historical_data: pd.DataFrame, agents_df: pd.DataFrame):
        """Train X-Learner on historical data"""
        print("Training Uplift Model (X-Learner)...")
        
        # Create feature matrix
        X = []
        y = []
        treatment = []  # 1 if high-skill match, 0 otherwise
        
        for _, row in historical_data.iterrows():
            agent = agents_df[agents_df['agent_id'] == row['agent_id']].iloc[0]
            
            # Mock customer from historical data
            customer = pd.Series({
                'skill_needed': row['customer_skill'],
                'priority': row['customer_priority'],
                'complexity': np.random.beta(2, 5),
                'wait_time': np.random.exponential(2),
                'channel': row['channel']
            })
            
            feat = self._featurize(customer, agent)
            X.append(feat)
            y.append(row['csat'])
            
            # Define treatment: high skill match = 1
            treatment.append(1 if row['skill_match'] > 0.6 else 0)
        
        X = np.array(X)
        y = np.array(y)
        treatment = np.array(treatment)
        
        # Step 1: Train outcome models for treated and control
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        self.mu1_model.fit(X_treated, y_treated)
        self.mu0_model.fit(X_control, y_control)
        
        # Step 2: Impute counterfactuals
        D1 = y_treated - self.mu0_model.predict(X_treated)
        D0 = self.mu1_model.predict(X_control) - y_control
        
        # Step 3: Train CATE models
        self.tau1_model.fit(X_treated, D1)
        self.tau0_model.fit(X_control, D0)
        
        self.is_trained = True
        print(f"✓ Uplift model trained on {len(X)} samples")
    
    def predict_uplift(self, customer: pd.Series, agent: pd.Series, 
                       exploration=False) -> Tuple[float, float]:
        """
        Predict CSAT uplift + uncertainty for (customer, agent) pair
        Returns: (uplift, uncertainty)
        """
        if not self.is_trained:
            # Cold start: use heuristic
            skill_match = agent[f'skill_{customer["skill_needed"]}']
            return skill_match * 0.3, 0.2
        
        X = self._featurize(customer, agent).reshape(1, -1)
        
        # Get predictions from both models
        tau1 = self.tau1_model.predict(X)[0]
        tau0 = self.tau0_model.predict(X)[0]
        
        # Average (propensity weighting would go here in production)
        uplift = 0.5 * (tau1 + tau0)
        
        # Uncertainty estimate (simplified - use bootstrap in production)
        uncertainty = np.abs(tau1 - tau0) / 2
        
        # Thompson sampling for exploration
        if exploration and config.THOMPSON_SAMPLING:
            uplift = np.random.normal(uplift, uncertainty)
        
        return uplift, uncertainty

class CapacityModel:
    """Model for predicting AHT and capacity constraints"""
    def __init__(self):
        self.aht_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.is_trained = False
    
    def train(self, historical_data: pd.DataFrame, agents_df: pd.DataFrame):
        """Train AHT prediction model"""
        print("Training Capacity Model (AHT Predictor)...")
        
        X = []
        y = []
        
        for _, row in historical_data.iterrows():
            agent = agents_df[agents_df['agent_id'] == row['agent_id']].iloc[0]
            
            features = [
                agent['avg_aht'],
                agent['experience_years'],
                row['skill_match'],
                config.CHANNELS.index(row['channel']),
                np.random.beta(2, 5)  # complexity proxy
            ]
            
            X.append(features)
            y.append(row['aht'])
        
        self.aht_model.fit(np.array(X), np.array(y))
        self.is_trained = True
        print("✓ Capacity model trained")
    
    def predict_aht(self, customer: pd.Series, agent: pd.Series) -> float:
        """Predict average handle time"""
        if not self.is_trained:
            return agent['avg_aht'] * (1.2 - agent[f'skill_{customer["skill_needed"]}'])
        
        features = np.array([[
            agent['avg_aht'],
            agent['experience_years'],
            agent[f'skill_{customer["skill_needed"]}'],
            config.CHANNELS.index(customer['channel']),
            customer['complexity']
        ]])
        
        return self.aht_model.predict(features)[0]
    
    def check_capacity(self, agent: pd.Series, channel: str) -> bool:
        """Check if agent can accept another interaction on this channel"""
        current_load = agent.get(f'load_{channel}', 0)
        max_capacity = config.CAPACITY_RULES[channel]
        
        # Cross-channel constraints
        for other_channel in config.CHANNELS:
            if other_channel != channel:
                other_load = agent.get(f'load_{other_channel}', 0)
                if other_load > 0:
                    cross_cap = config.CROSS_CHANNEL_CAPACITY[other_channel][channel]
                    max_capacity = min(max_capacity, cross_cap)
        
        return current_load < max_capacity

