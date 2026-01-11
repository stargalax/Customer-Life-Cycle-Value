"""
SQLite Database Layer - Enhanced for Production Deployment
Adds persistent storage, faster queries, and better scalability
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

class DatabaseManager:
    """
    SQLite database manager for CLV analytics
    Benefits: Persistent storage, SQL queries, better performance
    """
    
    def __init__(self, db_path='clv_analytics.db'):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database and tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()
        print(f"✅ Database initialized at {self.db_path}")
    
    def create_tables(self):
        """Create all necessary tables"""
        cursor = self.conn.cursor()
        
        # Main customers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                date_of_birth DATE,
                gender TEXT,
                email TEXT,
                phone_number TEXT,
                address_street TEXT,
                address_city TEXT,
                address_state TEXT,
                address_postal_code TEXT,
                address_country TEXT,
                policy_id TEXT,
                policy_type TEXT,
                policy_start_date DATE,
                policy_end_date DATE,
                annual_premium REAL,
                total_claims INTEGER,
                total_claim_amount REAL,
                customer_lifetime_value REAL,
                churn_risk_score REAL,
                is_churned BOOLEAN,
                last_interaction_date DATE,
                preferred_contact_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enriched features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_features (
                customer_id TEXT PRIMARY KEY,
                age INTEGER,
                policy_duration_days INTEGER,
                days_since_interaction INTEGER,
                claim_frequency REAL,
                avg_claim_amount REAL,
                recency_score REAL,
                frequency_score REAL,
                monetary_score REAL,
                rfm_score REAL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        ''')
        
        # Segments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_segments (
                customer_id TEXT PRIMARY KEY,
                segment_id INTEGER,
                segment_name TEXT,
                health_score REAL,
                engagement_score REAL,
                value_score REAL,
                retention_score REAL,
                predicted_churn_probability REAL,
                predicted_5yr_clv REAL,
                growth_category TEXT,
                assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                action_priority TEXT,
                action_type TEXT,
                action_description TEXT,
                reason TEXT,
                timeline TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        ''')
        
        # Analytics cache table (for performance)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_cache (
                cache_key TEXT PRIMARY KEY,
                cache_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_segment ON customer_segments(segment_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_churn_risk ON customers(churn_risk_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_policy_type ON customers(policy_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_country ON customers(address_country)')
        
        self.conn.commit()
        print("✅ All tables created with indexes")
    
    def load_csv_to_db(self, csv_path):
        """
        Load CSV data into SQLite database
        This is a one-time migration from CSV to DB
        """
        print(f"📥 Loading data from {csv_path}...")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Load into customers table
        df.to_sql('customers', self.conn, if_exists='replace', index=False)
        
        print(f"✅ Loaded {len(df)} records into database")
        return df
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results"""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    
    def get_customers(self, filters=None):
        """
        Get customers with optional filters
        Example: filters = {'segment_name': 'VIP Loyalists', 'churn_risk_score': '>0.5'}
        """
        query = '''
            SELECT c.*, cf.*, cs.*
            FROM customers c
            LEFT JOIN customer_features cf ON c.customer_id = cf.customer_id
            LEFT JOIN customer_segments cs ON c.customer_id = cs.customer_id
        '''
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str) and value.startswith('>'):
                    conditions.append(f"{key} > {value[1:]}")
                elif isinstance(value, str) and value.startswith('<'):
                    conditions.append(f"{key} < {value[1:]}")
                else:
                    conditions.append(f"{key} = '{value}'")
            
            query += ' WHERE ' + ' AND '.join(conditions)
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def save_features(self, features_df):
        """Save enriched features to database"""
        features_df.to_sql('customer_features', self.conn, if_exists='replace', index=False)
        print(f"✅ Saved {len(features_df)} feature records")
    
    def save_segments(self, segments_df):
        """Save customer segments to database"""
        segments_df.to_sql('customer_segments', self.conn, if_exists='replace', index=False)
        print(f"✅ Saved {len(segments_df)} segment assignments")
    
    def save_recommendations(self, recommendations):
        """Save recommendations to database"""
        cursor = self.conn.cursor()
        
        for rec in recommendations:
            for action in rec['actions']:
                cursor.execute('''
                    INSERT INTO recommendations 
                    (customer_id, action_priority, action_type, action_description, reason, timeline)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    rec['customer_id'],
                    action['priority'],
                    action['action'],
                    action['action'],
                    action['reason'],
                    action['timeline']
                ))
        
        self.conn.commit()
        print(f"✅ Saved recommendations")
    
    def get_analytics_summary(self):
        """Get analytics using SQL queries - much faster than pandas"""
        
        # Total customers
        total = self.execute_query("SELECT COUNT(*) FROM customers")[0][0]
        
        # Churn statistics
        churned = self.execute_query("SELECT COUNT(*) FROM customers WHERE is_churned = 1")[0][0]
        
        # CLV statistics
        clv_stats = self.execute_query('''
            SELECT 
                AVG(customer_lifetime_value) as avg_clv,
                SUM(customer_lifetime_value) as total_clv,
                MIN(customer_lifetime_value) as min_clv,
                MAX(customer_lifetime_value) as max_clv
            FROM customers
        ''')[0]
        
        # By policy type (SQL GROUP BY)
        policy_analysis = pd.read_sql_query('''
            SELECT 
                policy_type,
                COUNT(*) as customer_count,
                AVG(customer_lifetime_value) as avg_clv,
                SUM(customer_lifetime_value) as total_clv,
                AVG(annual_premium) as avg_premium,
                AVG(churn_risk_score) as avg_churn_risk
            FROM customers
            GROUP BY policy_type
        ''', self.conn)
        
        return {
            'total_customers': total,
            'churned_customers': churned,
            'churn_rate': round((churned / total) * 100, 2) if total > 0 else 0,
            'avg_clv': round(clv_stats[0], 2),
            'total_clv': round(clv_stats[1], 2),
            'min_clv': round(clv_stats[2], 2),
            'max_clv': round(clv_stats[3], 2),
            'by_policy_type': policy_analysis.to_dict('records')
        }
    
    def get_segment_summary(self):
        """Get segment summary using SQL"""
        segment_stats = pd.read_sql_query('''
            SELECT 
                cs.segment_name,
                COUNT(*) as customer_count,
                AVG(c.customer_lifetime_value) as avg_clv,
                AVG(c.churn_risk_score) as avg_churn_risk,
                AVG(cs.health_score) as avg_health_score,
                SUM(c.customer_lifetime_value) as total_clv
            FROM customer_segments cs
            JOIN customers c ON cs.customer_id = c.customer_id
            GROUP BY cs.segment_name
        ''', self.conn)
        
        return segment_stats.to_dict('records')
    
    def get_high_risk_customers(self, limit=50):
        """Get high-risk customers using SQL"""
        df = pd.read_sql_query(f'''
            SELECT 
                c.customer_id,
                c.first_name,
                c.last_name,
                c.email,
                c.churn_risk_score,
                c.customer_lifetime_value,
                cs.segment_name,
                cs.health_score
            FROM customers c
            LEFT JOIN customer_segments cs ON c.customer_id = cs.customer_id
            WHERE c.churn_risk_score > 0.7
            ORDER BY c.churn_risk_score DESC
            LIMIT {limit}
        ''', self.conn)
        
        return df.to_dict('records')
    
    def cache_analytics(self, key, value, expire_hours=24):
        """Cache expensive analytics for faster retrieval"""
        cursor = self.conn.cursor()
        expires_at = datetime.now().timestamp() + (expire_hours * 3600)
        
        cursor.execute('''
            INSERT OR REPLACE INTO analytics_cache (cache_key, cache_value, expires_at)
            VALUES (?, ?, ?)
        ''', (key, json.dumps(value, default=str), expires_at))
        
        self.conn.commit()
    
    def get_cached_analytics(self, key):
        """Retrieve cached analytics if not expired"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT cache_value FROM analytics_cache
            WHERE cache_key = ? AND expires_at > ?
        ''', (key, datetime.now().timestamp()))
        
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")


# Integration with existing modules
def integrate_database(csv_path='customer_data.csv'):
    """
    One-time setup to migrate from CSV to SQLite
    Run this once to set up the database
    """
    print("=== DATABASE MIGRATION ===")
    
    # Initialize database
    db = DatabaseManager()
    
    # Load CSV data
    df = db.load_csv_to_db(csv_path)
    
    # Run analytics pipeline
    from data_analysis import DataAnalyzer
    from predictive_analytics import PredictiveAnalytics
    from recommendations import RecommendationEngine
    
    print("\n=== Running Analytics Pipeline ===")
    analyzer = DataAnalyzer(csv_path)
    predictor = PredictiveAnalytics(analyzer.df)
    #predictor.customer_segmentation()
    #predictor.generate_customer_scores()
    predictor.customer_segmentation()
    predictor.churn_prediction_model()
    predictor.lifetime_value_prediction()
    predictor.generate_customer_scores()

    
    # Save enriched data to database
    print("\n=== Saving to Database ===")
    
    # Save features
    feature_cols = ['customer_id', 'age', 'policy_duration_days', 'days_since_interaction',
                   'claim_frequency', 'avg_claim_amount', 'recency_score', 'frequency_score',
                   'monetary_score', 'rfm_score']
    db.save_features(predictor.df[feature_cols])
    
    # Save segments
    segment_cols = ['customer_id', 'segment_id', 'segment_name', 'health_score',
                   'engagement_score', 'value_score', 'retention_score',
                   'predicted_churn_probability', 'predicted_5yr_clv', 'growth_category']
    db.save_segments(predictor.df[segment_cols])
    
    # Save recommendations
    recommender = RecommendationEngine(predictor.df)
    recommendations = recommender.generate_individual_recommendations()
    db.save_recommendations(recommendations)
    
    print("\n=== Migration Complete ===")
    print(f"Database ready at: {db.db_path}")
    print(f"Total customers: {len(df)}")
    
    return db


if __name__ == "__main__":
    # Run migration
    db = integrate_database('customer_data.csv')
    
    # Test queries
    print("\n=== Testing Database Queries ===")
    
    # Get summary
    summary = db.get_analytics_summary()
    print(f"\nAnalytics Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Get segments
    segments = db.get_segment_summary()
    print(f"\nSegment Summary:")
    for seg in segments:
        print(f"  {seg['segment_name']}: {seg['customer_count']} customers")
    
    # Get high-risk customers
    high_risk = db.get_high_risk_customers(limit=10)
    print(f"\nTop 10 High-Risk Customers:")
    for cust in high_risk:
        print(f"  {cust['first_name']} {cust['last_name']}: {cust['churn_risk_score']:.2%}")
    
    db.close()