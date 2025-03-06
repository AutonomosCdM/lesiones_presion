from typing import Dict, Any, List
from .models import RiskScoringAlgorithm, PatientRiskProfile
from sqlalchemy.orm import Session
from datetime import datetime
import logging

class RiskAssessmentService:
    """
    Service layer for risk assessment operations
    """
    def __init__(self, db_session: Session):
        """
        Initialize risk assessment service
        
        Args:
            db_session (Session): Database session
        """
        self.db = db_session
        self.risk_algorithm = RiskScoringAlgorithm()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def assess_patient_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive patient risk assessment
        
        Args:
            patient_data (dict): Patient's complete profile
        
        Returns:
            dict: Risk assessment results
        """
        # Calculate risk score
        risk_score = self.risk_algorithm.calculate_risk_score(patient_data)
        risk_category = self.risk_algorithm.risk_category(risk_score)
        
        # Prepare risk profile for database storage
        risk_profile = PatientRiskProfile(
            patient_id=patient_data.get('patient_id'),
            age=patient_data.get('age'),
            medical_history=patient_data.get('medical_history', []),
            current_conditions=patient_data.get('current_conditions', {}),
            mobility_score=patient_data.get('mobility_score', 0.5),
            nutrition_score=patient_data.get('nutrition_score', 0.5),
            skin_condition=patient_data.get('skin_condition', {}),
            risk_factors=patient_data.get('risk_factors', {}),
            calculated_risk_score=risk_score,
            last_assessment_date=datetime.now()
        )
        
        # Log assessment details
        self.logger.info(f"Risk Assessment for Patient {risk_profile.patient_id}")
        self.logger.info(f"Risk Score: {risk_score}")
        self.logger.info(f"Risk Category: {risk_category}")
        
        # Save to database
        try:
            self.db.add(risk_profile)
            self.db.commit()
            self.db.refresh(risk_profile)
        except Exception as e:
            self.logger.error(f"Error saving risk profile: {e}")
            self.db.rollback()
            raise
        
        return {
            'patient_id': risk_profile.patient_id,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'assessment_date': risk_profile.last_assessment_date
        }
    
    def get_patient_risk_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve patient's risk assessment history
        
        Args:
            patient_id (str): Unique patient identifier
        
        Returns:
            list: Historical risk assessments
        """
        try:
            risk_profiles = (
                self.db.query(PatientRiskProfile)
                .filter(PatientRiskProfile.patient_id == patient_id)
                .order_by(PatientRiskProfile.last_assessment_date.desc())
                .all()
            )
            
            return [
                {
                    'assessment_date': profile.last_assessment_date,
                    'risk_score': profile.calculated_risk_score,
                    'risk_category': self.risk_algorithm.risk_category(profile.calculated_risk_score)
                }
                for profile in risk_profiles
            ]
        except Exception as e:
            self.logger.error(f"Error retrieving risk history: {e}")
            raise
    
    def identify_high_risk_patients(self, risk_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify patients with high risk of pressure injuries
        
        Args:
            risk_threshold (float): Minimum risk score to be considered high risk
        
        Returns:
            list: High-risk patient profiles
        """
        try:
            high_risk_profiles = (
                self.db.query(PatientRiskProfile)
                .filter(PatientRiskProfile.calculated_risk_score >= risk_threshold)
                .order_by(PatientRiskProfile.calculated_risk_score.desc())
                .all()
            )
            
            return [
                {
                    'patient_id': profile.patient_id,
                    'risk_score': profile.calculated_risk_score,
                    'risk_category': self.risk_algorithm.risk_category(profile.calculated_risk_score),
                    'last_assessment_date': profile.last_assessment_date
                }
                for profile in high_risk_profiles
            ]
        except Exception as e:
            self.logger.error(f"Error identifying high-risk patients: {e}")
            raise
    
    def update_risk_assessment_weights(self, new_weights: Dict[str, float]):
        """
        Update risk assessment algorithm weights
        
        Args:
            new_weights (dict): New risk factor weights
        """
        try:
            # Validate weights
            if not all(0 <= weight <= 1 for weight in new_weights.values()):
                raise ValueError("Weights must be between 0 and 1")
            
            if abs(sum(new_weights.values()) - 1.0) > 1e-10:
                raise ValueError("Weights must sum to 1")
            
            # Update algorithm weights
            self.risk_algorithm.weights = new_weights
            self.logger.info("Risk assessment weights updated")
        except Exception as e:
            self.logger.error(f"Error updating risk weights: {e}")
            raise
