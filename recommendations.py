"""
Recommendations Engine - JPMorgan DART Project
Demonstrates: Problem-solving mindset, strategic thinking, business value delivery
"""

import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """
    Business rules engine for retention and growth strategies
    Skills: Solving unstructured problems, strategic focus, identifying opportunities
    """
    
    def __init__(self, df):
        """Initialize with segmented customer data"""
        self.df = df.copy()
        self.strategies = {}
    
    def generate_segment_strategies(self):
        """
        Create tailored retention strategies for each segment
        Skills: Strategic thinking, business goal focus
        """
        strategies = {
            'VIP Loyalists': {
                'priority': 'High',
                'objective': 'Maximize retention and lifetime value',
                'retention_strategies': [
                    {
                        'strategy': 'Exclusive VIP Program',
                        'description': 'Dedicated relationship manager with 24/7 access',
                        'expected_impact': 'Increase retention by 15-20%',
                        'cost': 'High',
                        'timeline': 'Immediate'
                    },
                    {
                        'strategy': 'Loyalty Rewards',
                        'description': 'Annual premium discounts (15-20%) and cashback programs',
                        'expected_impact': 'Reduce churn by 10%',
                        'cost': 'Medium',
                        'timeline': 'Quarterly'
                    },
                    {
                        'strategy': 'Early Product Access',
                        'description': 'Beta access to new insurance products and services',
                        'expected_impact': 'Increase cross-sell by 25%',
                        'cost': 'Low',
                        'timeline': 'Ongoing'
                    },
                    {
                        'strategy': 'Personalized Financial Planning',
                        'description': 'Complimentary annual financial health checkups',
                        'expected_impact': 'Increase policy value by 30%',
                        'cost': 'Medium',
                        'timeline': 'Annual'
                    }
                ],
                'communication': {
                    'channel': ['Phone', 'Email', 'In-person meetings'],
                    'frequency': 'Monthly',
                    'tone': 'Premium, exclusive, personalized'
                },
                'budget_allocation': 'Premium tier - $500-1000 per customer annually',
                'success_metrics': ['Retention rate >95%', 'NPS >80', 'Upsell rate >40%']
            },
            
            'Steady Contributors': {
                'priority': 'Medium-High',
                'objective': 'Increase engagement and policy consolidation',
                'retention_strategies': [
                    {
                        'strategy': 'Multi-Policy Bundling',
                        'description': 'Discounts for bundling 2+ policies (10-15% savings)',
                        'expected_impact': 'Increase policies per customer by 35%',
                        'cost': 'Low',
                        'timeline': 'Next renewal cycle'
                    },
                    {
                        'strategy': 'Automated Value Communication',
                        'description': 'Quarterly personalized reports showing coverage benefits',
                        'expected_impact': 'Reduce churn by 8%',
                        'cost': 'Low',
                        'timeline': 'Quarterly'
                    },
                    {
                        'strategy': 'Life Event Triggers',
                        'description': 'Proactive outreach during major life changes',
                        'expected_impact': 'Increase upsell by 20%',
                        'cost': 'Medium',
                        'timeline': 'Event-based'
                    },
                    {
                        'strategy': 'Referral Incentives',
                        'description': '$100-200 rewards for successful customer referrals',
                        'expected_impact': 'Acquire 0.3 new customers per existing',
                        'cost': 'Medium',
                        'timeline': 'Ongoing'
                    }
                ],
                'communication': {
                    'channel': ['Email', 'SMS', 'App notifications'],
                    'frequency': 'Bi-monthly',
                    'tone': 'Helpful, value-focused, professional'
                },
                'budget_allocation': 'Standard tier - $200-400 per customer annually',
                'success_metrics': ['Retention rate >85%', 'Policies per customer >1.5', 'Referral rate >15%']
            },
            
            'At-Risk Customers': {
                'priority': 'Critical',
                'objective': 'Immediate intervention to prevent churn',
                'retention_strategies': [
                    {
                        'strategy': 'Emergency Retention Campaign',
                        'description': 'Personal outreach within 48 hours of risk detection',
                        'expected_impact': 'Save 40-50% of at-risk customers',
                        'cost': 'High',
                        'timeline': 'Immediate (48hr SLA)'
                    },
                    {
                        'strategy': 'Special Retention Offers',
                        'description': 'Limited-time discounts (20-25%) or premium freezes',
                        'expected_impact': 'Reduce immediate churn by 35%',
                        'cost': 'High',
                        'timeline': 'Immediate'
                    },
                    {
                        'strategy': 'Exit Survey & Feedback',
                        'description': 'Understand pain points through structured interviews',
                        'expected_impact': 'Improve product by addressing top 3 issues',
                        'cost': 'Low',
                        'timeline': 'Within 7 days'
                    },
                    {
                        'strategy': 'Simplified Claims Process',
                        'description': 'Fast-track claims and dedicated support line',
                        'expected_impact': 'Increase satisfaction by 40%',
                        'cost': 'Medium',
                        'timeline': 'Immediate'
                    }
                ],
                'communication': {
                    'channel': ['Phone (primary)', 'SMS', 'Email'],
                    'frequency': 'Weekly until stabilized',
                    'tone': 'Urgent, empathetic, solution-focused'
                },
                'budget_allocation': 'High priority - $300-600 per customer (one-time)',
                'success_metrics': ['Win-back rate >45%', 'Complaint resolution <7 days', 'Satisfaction >7/10']
            },
            
            'Entry Level': {
                'priority': 'Medium',
                'objective': 'Nurture and upgrade to comprehensive coverage',
                'retention_strategies': [
                    {
                        'strategy': 'Education Program',
                        'description': 'Interactive content explaining policy benefits',
                        'expected_impact': 'Increase policy understanding by 60%',
                        'cost': 'Low',
                        'timeline': 'First 6 months'
                    },
                    {
                        'strategy': 'Upgrade Path',
                        'description': 'Clear roadmap to premium coverage with incentives',
                        'expected_impact': 'Upgrade 25% to higher tiers',
                        'cost': 'Low',
                        'timeline': '12-18 months'
                    },
                    {
                        'strategy': 'Digital-First Engagement',
                        'description': 'Mobile app with gamification and rewards',
                        'expected_impact': 'Increase app usage by 70%',
                        'cost': 'Medium',
                        'timeline': 'Ongoing'
                    },
                    {
                        'strategy': 'Peer Referral Program',
                        'description': '$50-100 incentives for bringing friends',
                        'expected_impact': 'Acquire 0.2 customers per existing',
                        'cost': 'Low',
                        'timeline': 'Ongoing'
                    }
                ],
                'communication': {
                    'channel': ['App', 'SMS', 'Email'],
                    'frequency': 'Monthly',
                    'tone': 'Educational, friendly, encouraging'
                },
                'budget_allocation': 'Cost-efficient - $100-200 per customer annually',
                'success_metrics': ['Retention rate >70%', 'Upgrade rate >25%', 'App engagement >50%']
            },
            
            'Growth Potential': {
                'priority': 'Medium-High',
                'objective': 'Accelerate value growth and cross-sell',
                'retention_strategies': [
                    {
                        'strategy': 'Targeted Upsell Campaigns',
                        'description': 'AI-driven recommendations for complementary products',
                        'expected_impact': 'Increase average premium by 40%',
                        'cost': 'Medium',
                        'timeline': 'Next 6 months'
                    },
                    {
                        'strategy': 'Life Stage Marketing',
                        'description': 'Trigger campaigns based on age, family status changes',
                        'expected_impact': 'Conversion rate >30%',
                        'cost': 'Medium',
                        'timeline': 'Event-based'
                    },
                    {
                        'strategy': 'Claims Excellence',
                        'description': 'White-glove service during claims',
                        'expected_impact': 'Increase NPS by 25 points',
                        'cost': 'Medium',
                        'timeline': 'Ongoing'
                    },
                    {
                        'strategy': 'Loyalty Points Program',
                        'description': 'Accumulate points, redeem for discounts',
                        'expected_impact': 'Increase retention by 12%',
                        'cost': 'Low',
                        'timeline': 'Quarterly'
                    }
                ],
                'communication': {
                    'channel': ['Email', 'App', 'SMS'],
                    'frequency': 'Bi-weekly',
                    'tone': 'Aspirational, growth-focused, personalized'
                },
                'budget_allocation': 'Standard tier - $250-450 per customer annually',
                'success_metrics': ['Revenue growth >40%', 'Cross-sell rate >35%', 'Retention >80%']
            }
        }
        
        self.strategies = strategies
        return strategies
    
    def generate_individual_recommendations(self):
        """
        Create personalized action items for each customer
        Skills: Attention to detail, solving problems independently
        """
        recommendations = []
        
        for idx, customer in self.df.iterrows():
            cust_rec = {
                'customer_id': customer['customer_id'],
                'name': f"{customer['first_name']} {customer['last_name']}",
                'segment': customer.get('segment_name', 'Unknown'),
                'actions': []
            }
            
            # High churn risk
            if customer['churn_risk_score'] > 0.7:
                cust_rec['actions'].append({
                    'priority': 'CRITICAL',
                    'action': 'Immediate retention call',
                    'reason': f"Churn risk: {customer['churn_risk_score']:.0%}",
                    'timeline': 'Within 48 hours'
                })
            
            # No recent interaction
            if pd.notna(customer.get('days_since_interaction')) and customer['days_since_interaction'] > 180:
                cust_rec['actions'].append({
                    'priority': 'HIGH',
                    'action': 'Re-engagement campaign',
                    'reason': f"No contact for {int(customer['days_since_interaction'])} days",
                    'timeline': 'This week'
                })
            
            # High value, no claims
            if customer['customer_lifetime_value'] > 30000 and customer['total_claims'] == 0:
                cust_rec['actions'].append({
                    'priority': 'MEDIUM',
                    'action': 'Cross-sell premium products',
                    'reason': 'High value, excellent claims history',
                    'timeline': 'Next quarter'
                })
            
            # Policy expiring soon
            if pd.notna(customer.get('policy_end_date')):
                try:
                    days_to_expiry = (pd.to_datetime(customer['policy_end_date']) - datetime.now()).days
                    if 0 < days_to_expiry < 60:
                        cust_rec['actions'].append({
                            'priority': 'HIGH',
                            'action': 'Renewal reminder with loyalty discount',
                            'reason': f"Policy expires in {days_to_expiry} days",
                            'timeline': 'Immediate'
                        })
                except:
                    pass
            
            # Young with growth potential
            if customer['age'] < 35 and customer['customer_lifetime_value'] < 20000:
                cust_rec['actions'].append({
                    'priority': 'LOW',
                    'action': 'Upgrade path communication',
                    'reason': 'Young customer with long-term potential',
                    'timeline': 'Next 6 months'
                })
            
            if len(cust_rec['actions']) > 0:
                recommendations.append(cust_rec)
        
        # Sort by number of actions
        recommendations.sort(key=lambda x: len(x['actions']), reverse=True)
        
        print(f"\n=== INDIVIDUAL RECOMMENDATIONS GENERATED ===")
        print(f"Total customers with action items: {len(recommendations)}")
        
        return recommendations
    
    def calculate_roi_projections(self):
        """
        Calculate expected ROI for retention strategies
        Skills: Business value delivery, strategic focus
        """
        roi_projections = {}
        
        for segment in self.df['segment_name'].unique():
            segment_data = self.df[self.df['segment_name'] == segment]
            
            total_customers = len(segment_data)
            avg_clv = segment_data['customer_lifetime_value'].mean()
            avg_churn_risk = segment_data['churn_risk_score'].mean()
            
            # Estimate costs and benefits per segment
            if segment == 'VIP Loyalists':
                cost_per_customer = 750
                retention_improvement = 0.10
                upsell_rate = 0.40
            elif segment == 'Steady Contributors':
                cost_per_customer = 300
                retention_improvement = 0.08
                upsell_rate = 0.20
            elif segment == 'At-Risk Customers':
                cost_per_customer = 450
                retention_improvement = 0.35
                upsell_rate = 0.05
            elif segment == 'Entry Level':
                cost_per_customer = 150
                retention_improvement = 0.15
                upsell_rate = 0.25
            else:  # Growth Potential
                cost_per_customer = 350
                retention_improvement = 0.12
                upsell_rate = 0.35
            
            # Calculate ROI
            total_investment = cost_per_customer * total_customers
            retained_value = avg_clv * avg_churn_risk * retention_improvement * total_customers
            upsell_value = avg_clv * 0.3 * upsell_rate * total_customers
            total_benefit = retained_value + upsell_value
            net_benefit = total_benefit - total_investment
            roi_percentage = (net_benefit / total_investment) * 100 if total_investment > 0 else 0
            
            roi_projections[segment] = {
                'total_customers': int(total_customers),
                'investment': round(float(total_investment), 2),
                'expected_benefit': round(float(total_benefit), 2),
                'net_benefit': round(float(net_benefit), 2),
                'roi_percentage': round(float(roi_percentage), 2),
                'payback_period_months': round(12 / (roi_percentage / 100)) if roi_percentage > 0 else 'N/A'
            }
        
        print("\n=== ROI PROJECTIONS ===")
        for segment, proj in roi_projections.items():
            print(f"\n{segment}:")
            print(f"  Investment: ${proj['investment']:,.0f}")
            print(f"  Expected Benefit: ${proj['expected_benefit']:,.0f}")
            print(f"  ROI: {proj['roi_percentage']:.1f}%")
        
        return roi_projections
    
    def export_recommendations(self, output_path='recommendations_output.json'):
        """Export all recommendations for dashboard"""
        individual_recs = self.generate_individual_recommendations()
        roi = self.calculate_roi_projections()
        
        output = {
            'segment_strategies': self.strategies,
            'individual_recommendations': individual_recs[:50],
            'roi_projections': roi,
            'summary': {
                'total_segments': len(self.strategies),
                'customers_with_actions': len(individual_recs),
                'estimated_total_roi': sum(proj['net_benefit'] for proj in roi.values())
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n=== RECOMMENDATIONS EXPORTED TO {output_path} ===")
        return output


# Example usage
if __name__ == "__main__":
    import sys
    try:
        from data_analysis import DataAnalyzer
        from predictive_analytics import PredictiveAnalytics
        
        analyzer = DataAnalyzer('customer_data.csv')
        predictor = PredictiveAnalytics(analyzer.df)
        predictor.customer_segmentation()
        predictor.generate_customer_scores()
        
        recommender = RecommendationEngine(predictor.df)
        strategies = recommender.generate_segment_strategies()
        recommender.export_recommendations()
        
        print("\n=== RECOMMENDATIONS ENGINE COMPLETE ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)