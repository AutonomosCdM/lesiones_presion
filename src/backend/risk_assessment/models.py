from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, Any, List
import numpy as np

Base = declarative_base()

class PatientRiskProfile(Base):
    """
    SQLAlchemy model for storing patient risk profiles
    """
    __tablename__ = 'patient_risk_profiles'

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)
    age = Column(Integer)
    medical_history = Column(JSON)
    current_conditions = Column(JSON)
    mobility_score = Column(Float)
    nutrition_score = Column(Float)
    skin_condition = Column(JSON)
    risk_factors = Column(JSON)
    calculated_risk_score = Column(Float)
    last_assessment_date = Column(DateTime)

class RiskScoringAlgorithm:
    """
    Risk assessment algorithm for Pressure Injury prediction
    """
    def __init__(self, risk_weights: Dict[str, float] = None):
        """
        Initialize risk scoring algorithm
        
        Args:
            risk_weights (dict): Customizable risk factor weights
        """
        # Default risk factor weights
        self.default_weights = {
            'age': 0.15,
            'mobility': 0.25,
            'nutrition': 0.20,
            'skin_condition': 0.15,
            'medical_history': 0.25
        }
        
        # Override default weights if provided
        self.weights = risk_weights or self.default_weights
    
    def calculate_age_risk(self, age: int) -> float:
        """
        Calculate risk based on patient age
        
        Args:
            age (int): Patient's age
        
        Returns:
            float: Age-related risk score
        """
        if age < 30:
            return 0.1
        elif 30 <= age < 50:
            return 0.3
        elif 50 <= age < 70:
            return 0.6
        else:
            return 0.9
    
    def calculate_mobility_risk(self, mobility_score: float) -> float:
        """
        Calculate risk based on patient mobility
        
        Args:
            mobility_score (float): Patient's mobility score (0-1)
        
        Returns:
            float: Mobility-related risk score
        """
        return 1.0 - mobility_score
    
    def calculate_nutrition_risk(self, nutrition_score: float) -> float:
        """
        Calculate risk based on patient nutrition
        
        Args:
            nutrition_score (float): Patient's nutrition score (0-1)
        
        Returns:
            float: Nutrition-related risk score
        """
        return 1.0 - nutrition_score
    
    def calculate_skin_condition_risk(self, skin_condition: Dict[str, Any]) -> float:
        """
        Calculate risk based on skin condition
        
        Args:
            skin_condition (dict): Patient's skin condition details
        
        Returns:
            float: Skin condition-related risk score
        """
        risk_factors = [
            skin_condition.get('moisture', 0),
            skin_condition.get('friction', 0),
            skin_condition.get('existing_wounds', 0)
        ]
        return np.mean(risk_factors)
    
    def calculate_medical_history_risk(self, medical_history: List[str]) -> float:
        """
        Calculate risk based on medical history
        
        Args:
            medical_history (list): Patient's medical history conditions
        
        Returns:
            float: Medical history-related risk score
        """
        risk_conditions = {
            'diabetes': 0.3,
            'cardiovascular_disease': 0.2,
            'neurological_disorder': 0.25,
            'immobility': 0.4,
            'malnutrition': 0.35
        }
        
        history_risk = [
            risk_conditions.get(condition, 0.1) 
            for condition in medical_history
        ]
        
        return np.max(history_risk) if history_risk else 0.1
    
    def calculate_risk_score(self, patient_profile: Dict[str, Any]) -> float:
        """
        Calculate comprehensive risk score
        
        Args:
            patient_profile (dict): Patient's complete risk profile
        
        Returns:
            float: Comprehensive risk score (0-1)
        """
        # Calculate individual risk components
        age_risk = self.calculate_age_risk(patient_profile.get('age', 50))
        mobility_risk = self.calculate_mobility_risk(
            patient_profile.get('mobility_score', 0.5)
        )
        nutrition_risk = self.calculate_nutrition_risk(
            patient_profile.get('nutrition_score', 0.5)
        )
        skin_risk = self.calculate_skin_condition_risk(
            patient_profile.get('skin_condition', {})
        )
        medical_history_risk = self.calculate_medical_history_risk(
            patient_profile.get('medical_history', [])
        )
        
        # Weighted risk calculation
        risk_components = {
            'age': age_risk,
            'mobility': mobility_risk,
            'nutrition': nutrition_risk,
            'skin_condition': skin_risk,
            'medical_history': medical_history_risk
        }
        
        # Calculate weighted risk score
        risk_score = sum(
            risk_components[factor] * self.weights[factor] 
            for factor in self.weights
        )
        
        return min(max(risk_score, 0), 1)  # Clamp between 0 and 1
    
    def risk_category(self, risk_score: float) -> str:
        """
        Categorize risk level
        
        Args:
            risk_score (float): Calculated risk score
        
        Returns:
            str: Risk category
        """
        if risk_score < 0.2:
            return 'Low Risk'
        elif 0.2 <= risk_score < 0.5:
            return 'Moderate Risk'
        elif 0.5 <= risk_score < 0.8:
            return 'High Risk'
        else:
            return 'Very High Risk'
