"""
Predictive Analytics Module - JPMorgan DART Project
Demonstrates: Python ML techniques, clustering, predictive modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalytics:
    """
    Machine Learning engine for customer segmentation and churn prediction
    Skills: Python, data analysis, innovation in data methods
    """
    
    def __init__(self, df):
        """Initialize with preprocessed dataframe"""
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.segments = None
        self.churn_model = None
    
    # def create_rfm_features(self):
    #     """
    #     RFM Analysis (Recency, Frequency, Monetary)
    #     Innovation: Adapting marketing analytics for insurance
    #     """
    #     # Recency: Days since last interaction
    #     self.df['recency_score'] = pd.qcut(
    #         self.df['days_since_interaction'].fillna(self.df['days_since_interaction'].max()),
    #         q=5, labels=[5, 4, 3, 2, 1], duplicates='drop'
    #     ).astype(float)
        
    #     # Frequency: Claims frequency
    #     self.df['frequency_score'] = pd.qcut(
    #         self.df['claim_frequency'].fillna(0),
    #         q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    #     ).astype(float)
        
    #     # Monetary: Customer Lifetime Value
    #     self.df['monetary_score'] = pd.qcut(
    #         self.df['customer_lifetime_value'],
    #         q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    #     ).astype(float)
        
    #     # Combined RFM Score
    #     self.df['rfm_score'] = (
    #         self.df['recency_score'] + 
    #         self.df['frequency_score'] + 
    #         self.df['monetary_score']
    #     ) / 3
        
    #     print("=== RFM FEATURES CREATED ===")
    #     print(f"Recency, Frequency, Monetary scores calculated")
    def create_rfm_features(self):

        def safe_qcut(series, q, ascending=True):
            series = series.fillna(series.max() if ascending else series.min())
            ranked = series.rank(method="first", ascending=ascending)

            try:
                bins = pd.qcut(ranked, q=q, duplicates='drop')
                n_bins = bins.cat.categories.size
                labels = list(range(1, n_bins + 1))
                return pd.qcut(ranked, q=q, labels=labels, duplicates='drop').astype(float)
            except ValueError:
                # Fallback: single bin
                return pd.Series([1.0] * len(series), index=series.index)

        # Recency (lower days = better)
        self.df['recency_score'] = safe_qcut(
            self.df['days_since_interaction'],
            q=5,
            ascending=True
        )

        # Frequency (higher = better)
        self.df['frequency_score'] = safe_qcut(
            self.df['claim_frequency'],
            q=5,
            ascending=False
        )

        # Monetary (higher = better)
        self.df['monetary_score'] = safe_qcut(
            self.df['customer_lifetime_value'],
            q=5,
            ascending=False
        )

        # Combined RFM Score
        self.df['rfm_score'] = (
            self.df['recency_score'] +
            self.df['frequency_score'] +
            self.df['monetary_score']
        ) / 3

        print("=== RFM FEATURES CREATED (SAFE) ===")

        
    def customer_segmentation(self, n_clusters=5):
            """
            K-Means clustering for customer segmentation
            Skills: Python ML, pattern interpretation
            """
            print("\n=== CUSTOMER SEGMENTATION ===")
            
            # Create RFM features first
            self.create_rfm_features()
            
            # Select features for clustering
            feature_cols = [
                'customer_lifetime_value',
                'annual_premium',
                'churn_risk_score',
                'age',
                'total_claims',
                'policy_duration_days',
                'claim_frequency',
                'rfm_score'
            ]
            
            # Prepare data
            X = self.df[feature_cols].fillna(0)
            
            # Standardize features (important for K-means)
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['segment_id'] = kmeans.fit_predict(X_scaled)
            
            # Analyze cluster characteristics
            segment_profiles = self.df.groupby('segment_id').agg({
                'customer_lifetime_value': ['mean', 'count'],
                'annual_premium': 'mean',
                'churn_risk_score': 'mean',
                'age': 'mean',
                'total_claims': 'mean',
                'is_churned': lambda x: (x.sum() / len(x) * 100),
                'rfm_score': 'mean'
            }).round(2)
            
            print("\nSegment Profiles:")
            print(segment_profiles)
            
            # Assign meaningful segment names based on characteristics
            self.assign_segment_names()
            
            return segment_profiles
        
    def assign_segment_names(self):
        """
        Assign business-friendly names to segments
        Skills: Strategic thinking, business focus
        """
        segment_mapping = {}
        
        for seg_id in self.df['segment_id'].unique():
            segment_data = self.df[self.df['segment_id'] == seg_id]
            
            avg_clv = segment_data['customer_lifetime_value'].mean()
            avg_risk = segment_data['churn_risk_score'].mean()
            avg_premium = segment_data['annual_premium'].mean()
            churn_rate = segment_data['is_churned'].mean()
            
            # Decision tree logic for naming
            if avg_clv > 50000 and avg_risk < 0.3:
                name = 'VIP Loyalists'
            elif avg_clv > 30000 and avg_risk < 0.5:
                name = 'Steady Contributors'
            elif avg_risk > 0.7 or churn_rate > 0.5:
                name = 'At-Risk Customers'
            elif avg_clv < 10000 and avg_premium < 2000:
                name = 'Entry Level'
            else:
                name = 'Growth Potential'
            
            segment_mapping[seg_id] = name
        
        self.df['segment_name'] = self.df['segment_id'].map(segment_mapping)
        
        print("\n=== SEGMENT NAMES ASSIGNED ===")
        for seg_id, name in segment_mapping.items():
            count = len(self.df[self.df['segment_id'] == seg_id])
            print(f"Segment {seg_id}: {name} ({count} customers)")
        
        self.segments = segment_mapping
    
    def churn_prediction_model(self):
        """
        Random Forest classifier for churn prediction
        Skills: Python ML, solving unstructured problems
        """
        print("\n=== CHURN PREDICTION MODEL ===")
        
        # Features for prediction
        feature_cols = [
            'customer_lifetime_value',
            'annual_premium',
            'churn_risk_score',
            'age',
            'total_claims',
            'total_claim_amount',
            'policy_duration_days',
            'days_since_interaction',
            'claim_frequency',
            'rfm_score'
        ]
        
        # Prepare data
        X = self.df[feature_cols].fillna(0)
        y = self.df['is_churned']
        
        # Check if we have enough samples for both classes
        if y.sum() < 2 or len(y) - y.sum() < 2:
            print("Warning: Not enough samples for both classes. Using simplified prediction.")
            # Use churn_risk_score as proxy
            self.df['predicted_churn_probability'] = self.df['churn_risk_score']
            return {
                'model': None,
                'feature_importance': [],
                'test_accuracy': 0.0
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        
        # Evaluation
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        # Store model
        self.churn_model = rf_model
        
        # Add predictions to dataframe
        self.df['predicted_churn_probability'] = rf_model.predict_proba(X)[:, 1]
        
        return {
            'model': rf_model,
            'feature_importance': feature_importance.to_dict('records'),
            'test_accuracy': rf_model.score(X_test, y_test)
        }
    
    def lifetime_value_prediction(self):
        """
        Predict future CLV based on current behavior
        Skills: Innovation, strategic focus
        """
        print("\n=== CLV PREDICTION ===")
        
        # Calculate CLV growth rate
        self.df['clv_per_year'] = self.df['customer_lifetime_value'] / (
            self.df['policy_duration_days'] / 365
        ).replace(0, 1)  # Avoid division by zero
        
        # Predict 5-year CLV
        self.df['predicted_5yr_clv'] = self.df.apply(
            lambda row: row['customer_lifetime_value'] + 
                       (row['clv_per_year'] * 5 * (1 - row['churn_risk_score'])),
            axis=1
        )
        
        # Categorize growth potential
        growth_diff = self.df['predicted_5yr_clv'] - self.df['customer_lifetime_value']
        self.df['growth_category'] = pd.cut(
            growth_diff,
            bins=[-float('inf'), 0, 10000, 50000, float('inf')],
            labels=['Declining', 'Stable', 'Growing', 'High Growth']
        )
        
        growth_summary = self.df['growth_category'].value_counts()
        print("\nGrowth Potential Distribution:")
        print(growth_summary)
        
        return growth_summary.to_dict()
    
    def generate_customer_scores(self):
        """
        Generate comprehensive customer scores
        Skills: Assembling data, building reports
        """
        # Engagement Score (0-100)
        max_days = self.df['days_since_interaction'].max()
        if pd.isna(max_days) or max_days == 0:
            max_days = 365
        
        self.df['engagement_score'] = 100 - (
            self.df['days_since_interaction'].fillna(max_days) / max_days * 100
        ).clip(0, 100)
        
        # Value Score (0-100) - normalized CLV
        clv_min = self.df['customer_lifetime_value'].min()
        clv_max = self.df['customer_lifetime_value'].max()
        if clv_max == clv_min:
            self.df['value_score'] = 50.0
        else:
            self.df['value_score'] = (
                (self.df['customer_lifetime_value'] - clv_min) / (clv_max - clv_min) * 100
            )
        
        # Retention Score (0-100) - inverse of churn risk
        self.df['retention_score'] = (1 - self.df['churn_risk_score']) * 100
        
        # Overall Health Score (weighted average)
        self.df['health_score'] = (
            self.df['engagement_score'] * 0.3 +
            self.df['value_score'] * 0.4 +
            self.df['retention_score'] * 0.3
        )
        
        print("\n=== CUSTOMER SCORES GENERATED ===")
        print(f"Engagement, Value, Retention, and Health scores calculated")
    
    def export_predictions(self, output_path='predictions_output.json'):
        """
        Export all predictions and segments for dashboard
        """
        output = {
            'segments': self.df.groupby('segment_name').agg({
                'customer_id': 'count',
                'customer_lifetime_value': 'mean',
                'churn_risk_score': 'mean',
                'health_score': 'mean',
                'predicted_5yr_clv': 'mean'
            }).round(2).to_dict(),
            'segment_list': self.df[['customer_id', 'segment_name', 'segment_id']].to_dict('records'),
            'high_risk_customers': self.df[
                self.df['churn_risk_score'] > 0.7
            ][['customer_id', 'first_name', 'last_name', 'churn_risk_score', 'segment_name']].to_dict('records'),
            'top_value_customers': self.df.nlargest(20, 'customer_lifetime_value')[
                ['customer_id', 'first_name', 'last_name', 'customer_lifetime_value', 'segment_name']
            ].to_dict('records'),
            'growth_potential': self.df[
                (self.df['growth_category'] == 'High Growth') & 
                (self.df['churn_risk_score'] < 0.5)
            ][['customer_id', 'first_name', 'last_name', 'predicted_5yr_clv', 'segment_name']].to_dict('records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n=== PREDICTIONS EXPORTED TO {output_path} ===")
        return output


# Example usage
if __name__ == "__main__":
    # Load data (assumes you've run data_analysis.py first)
    import sys
    try:
        from data_analysis import DataAnalyzer
        
        analyzer = DataAnalyzer('customer_data.csv')
        
        # Initialize predictive analytics
        predictor = PredictiveAnalytics(analyzer.df)
        
        # Perform customer segmentation
        segment_profiles = predictor.customer_segmentation(n_clusters=5)
        
        # Build churn prediction model
        churn_results = predictor.churn_prediction_model()
        
        # Predict lifetime value
        clv_predictions = predictor.lifetime_value_prediction()
        
        # Generate customer scores
        predictor.generate_customer_scores()
        
        # Export all predictions
        predictor.export_predictions()
        
        print("\n=== PREDICTIVE ANALYTICS COMPLETE ===")
        print(f"Total customers analyzed: {len(predictor.df)}")
        print(f"Segments created: {len(predictor.segments)}")
        if churn_results['test_accuracy'] > 0:
            print(f"Churn model accuracy: {churn_results['test_accuracy']:.2%}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure data_analysis.py is in the same directory and customer_data.csv exists.")
        sys.exit(1)