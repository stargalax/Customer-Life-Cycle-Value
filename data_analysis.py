"""
Data Analysis Module - JPMorgan DART Project
Demonstrates: SQL operations, data transformation, statistical analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

class DataAnalyzer:
    """
    Core data analysis engine using pandas for SQL-like operations
    Skills: Data collection, organization, pattern analysis
    """
    
    def __init__(self, csv_filepath):
        """Load and validate data with attention to detail"""
        self.df = pd.read_csv(csv_filepath)
        self.validate_data()
        self.transform_data()
    
    def validate_data(self):
        """
        Data validation - ensuring accuracy and completeness
        Identifies missing values, data types, and outliers
        """
        print("=== DATA VALIDATION REPORT ===")
        print(f"Total Records: {len(self.df)}")
        print(f"Total Columns: {len(self.df.columns)}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nData Types:\n{self.df.dtypes}")
        
        # Check for duplicates
        duplicates = self.df.duplicated(subset=['customer_id']).sum()
        print(f"\nDuplicate Customer IDs: {duplicates}")
    
    def transform_data(self):
        """
        Data transformation and feature engineering
        Skills: Managing and transforming data
        """
        # Convert date columns
        self.df['date_of_birth'] = pd.to_datetime(self.df['date_of_birth'])
        self.df['policy_start_date'] = pd.to_datetime(self.df['policy_start_date'])
        self.df['policy_end_date'] = pd.to_datetime(self.df['policy_end_date'])
        self.df['last_interaction_date'] = pd.to_datetime(self.df['last_interaction_date'], errors='coerce')
        
        # Convert boolean
        #self.df['is_churned'] = self.df['is_churned'].map({'TRUE': True, 'FALSE': False})
        self.df['is_churned'] = self.df['is_churned'].astype(str).str.upper().map({
            'TRUE': True, 'FALSE': False
        }).fillna(False)

        
        # Feature engineering
        current_date = datetime.now()
        self.df['age'] = (current_date - self.df['date_of_birth']).dt.days // 365
        self.df['policy_duration_days'] = (self.df['policy_end_date'] - self.df['policy_start_date']).dt.days
        self.df['days_since_interaction'] = (current_date - self.df['last_interaction_date']).dt.days
        
        # Calculate claim frequency
        self.df['claim_frequency'] = self.df['total_claims'] / (self.df['policy_duration_days'] / 365)
        self.df['claim_frequency'] = self.df['claim_frequency'].fillna(0)
        
        # Average claim amount
        self.df['avg_claim_amount'] = np.where(
            self.df['total_claims'] > 0,
            self.df['total_claim_amount'] / self.df['total_claims'],
            0
        )
        
        print("\n=== FEATURE ENGINEERING COMPLETE ===")
        print(f"New Features Added: age, policy_duration_days, days_since_interaction, claim_frequency, avg_claim_amount")
    
    def get_descriptive_statistics(self):
        """
        SQL-like SELECT with aggregations
        Skills: Analyzing complex data sets
        """
        numeric_cols = ['annual_premium', 'total_claims', 'total_claim_amount', 
                       'customer_lifetime_value', 'churn_risk_score', 'age']
        
        stats = self.df[numeric_cols].describe().round(2)
        
        # Additional metrics
        stats.loc['sum'] = self.df[numeric_cols].sum()
        stats.loc['variance'] = self.df[numeric_cols].var()
        
        return stats
    
    def group_analysis(self):
        """
        SQL GROUP BY operations
        Skills: Identifying trends and patterns
        """
        analyses = {}
        
        # Analysis 1: Policy Type Performance
        # SQL equivalent: SELECT policy_type, COUNT(*), AVG(customer_lifetime_value), SUM(annual_premium)
        #                 FROM customers GROUP BY policy_type
        analyses['by_policy_type'] = self.df.groupby('policy_type').agg({
            'customer_id': 'count',
            'customer_lifetime_value': ['mean', 'sum', 'median'],
            'annual_premium': ['mean', 'sum'],
            'total_claim_amount': 'sum',
            'churn_risk_score': 'mean',
            'is_churned': lambda x: (x.sum() / len(x) * 100)
        }).round(2)
        
        # Analysis 2: Geographic Distribution
        # SQL: SELECT address_country, COUNT(*), AVG(customer_lifetime_value)
        #      FROM customers GROUP BY address_country
        analyses['by_country'] = self.df.groupby('address_country').agg({
            'customer_id': 'count',
            'customer_lifetime_value': 'mean',
            'annual_premium': 'mean',
            'churn_risk_score': 'mean'
        }).round(2)
        
        # Analysis 3: Age Demographics
        # Create age bins
        self.df['age_group'] = pd.cut(
            self.df['age'], 
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56+']
        )
        
        analyses['by_age_group'] = self.df.groupby('age_group').agg({
            'customer_id': 'count',
            'customer_lifetime_value': 'mean',
            'churn_risk_score': 'mean',
            'is_churned': lambda x: (x.sum() / len(x) * 100)
        }).round(2)
        
        # Analysis 4: Gender Analysis
        analyses['by_gender'] = self.df.groupby('gender').agg({
            'customer_id': 'count',
            'customer_lifetime_value': ['mean', 'sum'],
            'annual_premium': 'mean',
            'total_claims': 'sum'
        }).round(2)
        
        # Analysis 5: Churn Analysis
        analyses['churn_analysis'] = self.df.groupby('is_churned').agg({
            'customer_id': 'count',
            'customer_lifetime_value': 'mean',
            'annual_premium': 'mean',
            'total_claims': 'mean',
            'days_since_interaction': 'mean'
        }).round(2)
        
        return analyses
    
    def identify_risks_and_opportunities(self):
        """
        Identify risks and opportunities to unlock value
        Skills: Strategic problem solving, business focus
        """
        risks = []
        opportunities = []
        
        # Risk 1: High churn risk customers
        high_churn = self.df[self.df['churn_risk_score'] > 0.7]
        total_at_risk_clv = high_churn['customer_lifetime_value'].sum()
        risks.append({
            'category': 'High Churn Risk',
            'count': len(high_churn),
            'potential_revenue_loss': total_at_risk_clv,
            'severity': 'CRITICAL',
            'action': 'Immediate retention campaign required'
        })
        
        # Risk 2: Customers with no recent interaction
        no_interaction = self.df[self.df['days_since_interaction'] > 180]
        risks.append({
            'category': 'Low Engagement',
            'count': len(no_interaction),
            'potential_revenue_loss': no_interaction['customer_lifetime_value'].sum(),
            'severity': 'HIGH',
            'action': 'Re-engagement campaign needed'
        })
        
        # Opportunity 1: High value, low claims customers
        upsell_candidates = self.df[
            (self.df['customer_lifetime_value'] > self.df['customer_lifetime_value'].median()) &
            (self.df['total_claims'] == 0) &
            (self.df['churn_risk_score'] < 0.3)
        ]
        opportunities.append({
            'category': 'Upsell/Cross-sell',
            'count': len(upsell_candidates),
            'potential_revenue': upsell_candidates['annual_premium'].sum() * 0.3,  # 30% upsell potential
            'action': 'Target for premium product offerings'
        })
        
        # Opportunity 2: Young customers with growth potential
        growth_segment = self.df[
            (self.df['age'] < 35) &
            (self.df['customer_lifetime_value'] < self.df['customer_lifetime_value'].median())
        ]
        opportunities.append({
            'category': 'Future Growth',
            'count': len(growth_segment),
            'potential_revenue': growth_segment['annual_premium'].sum() * 2,  # 2x growth over 5 years
            'action': 'Long-term relationship building'
        })
        
        return {'risks': risks, 'opportunities': opportunities}
    
    def export_for_dashboard(self, output_path='analytics_output.json'):
        """
        Assemble data and build reports for business partners
        Skills: Report building, stakeholder communication
        """
        output = {
            'summary_statistics': self.get_descriptive_statistics().to_dict(),
            'group_analyses': {k: v.to_dict() for k, v in self.group_analysis().items()},
            'risks_opportunities': self.identify_risks_and_opportunities(),
            'data_preview': self.df.head(20).to_dict('records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n=== ANALYSIS EXPORTED TO {output_path} ===")
        return output


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your CSV file
    analyzer = DataAnalyzer('customer_data.csv')
    
    # Get descriptive statistics
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(analyzer.get_descriptive_statistics())
    
    # Perform group analyses
    print("\n=== GROUP ANALYSES ===")
    analyses = analyzer.group_analysis()
    for name, result in analyses.items():
        print(f"\n{name.upper()}:")
        print(result)
    
    # Identify risks and opportunities
    print("\n=== RISKS & OPPORTUNITIES ===")
    ro = analyzer.identify_risks_and_opportunities()
    print(json.dumps(ro, indent=2, default=str))
    
    # Export for dashboard
    analyzer.export_for_dashboard()