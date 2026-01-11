"""
Production API with SQLite Database
Optimized for deployment on Render/Railway with persistent storage
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sqlite3
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database configuration
DB_PATH = os.environ.get('DB_PATH', 'clv_analytics.db')

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def dict_from_row(row):
    """Convert sqlite3.Row to dictionary"""
    return dict(zip(row.keys(), row))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'message': 'CLV Analytics API with SQLite is running',
            'database': 'connected',
            'total_customers': count,
            'version': '2.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """
    Get analytics summary using optimized SQL queries
    Much faster than pandas operations
    """
    try:
        conn = get_db_connection()
        
        # Total customers and churn
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        total_customers = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM customers WHERE is_churned = 1")
        churned_customers = cursor.fetchone()[0]
        
        # CLV statistics
        cursor.execute('''
            SELECT 
                AVG(customer_lifetime_value) as avg_clv,
                SUM(customer_lifetime_value) as total_clv,
                MIN(customer_lifetime_value) as min_clv,
                MAX(customer_lifetime_value) as max_clv,
                AVG(annual_premium) as avg_premium
            FROM customers
        ''')
        stats = cursor.fetchone()
        
        # By policy type
        cursor.execute('''
            SELECT 
                policy_type,
                COUNT(*) as count,
                AVG(customer_lifetime_value) as avg_clv,
                SUM(customer_lifetime_value) as total_clv,
                AVG(annual_premium) as avg_premium,
                AVG(churn_risk_score) as avg_churn_risk
            FROM customers
            GROUP BY policy_type
        ''')
        policy_data = [dict_from_row(row) for row in cursor.fetchall()]
        
        # By country
        cursor.execute('''
            SELECT 
                address_country,
                COUNT(*) as count,
                AVG(customer_lifetime_value) as avg_clv,
                AVG(annual_premium) as avg_premium
            FROM customers
            GROUP BY address_country
            ORDER BY count DESC
        ''')
        country_data = [dict_from_row(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'total_customers': total_customers,
            'churned_customers': churned_customers,
            'churn_rate': round((churned_customers / total_customers) * 100, 2) if total_customers > 0 else 0,
            'avg_clv': round(stats[0] or 0, 2),
            'total_clv': round(stats[1] or 0, 2),
            'min_clv': round(stats[2] or 0, 2),
            'max_clv': round(stats[3] or 0, 2),
            'avg_premium': round(stats[4] or 0, 2),
            'by_policy_type': policy_data,
            'by_country': country_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/segments', methods=['GET'])
def get_segments():
    """Get customer segments with profiles"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Segment summary
        cursor.execute('''
            SELECT 
                cs.segment_name,
                COUNT(*) as customer_count,
                AVG(c.customer_lifetime_value) as avg_clv,
                SUM(c.customer_lifetime_value) as total_clv,
                AVG(c.churn_risk_score) as avg_churn_risk,
                AVG(cs.health_score) as avg_health_score,
                AVG(cf.age) as avg_age
            FROM customer_segments cs
            JOIN customers c ON cs.customer_id = c.customer_id
            LEFT JOIN customer_features cf ON cs.customer_id = cf.customer_id
            GROUP BY cs.segment_name
        ''')
        segment_data = [dict_from_row(row) for row in cursor.fetchall()]
        
        # Get sample customers from each segment
        segment_customers = {}
        for segment in segment_data:
            cursor.execute('''
                SELECT 
                    c.customer_id,
                    c.first_name,
                    c.last_name,
                    c.customer_lifetime_value,
                    c.churn_risk_score,
                    cs.health_score
                FROM customers c
                JOIN customer_segments cs ON c.customer_id = cs.customer_id
                WHERE cs.segment_name = ?
                LIMIT 20
            ''', (segment['segment_name'],))
            segment_customers[segment['segment_name']] = [dict_from_row(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'segments': segment_data,
            'segment_customers': segment_customers
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/segments/<segment_name>', methods=['GET'])
def get_segment_details(segment_name):
    """Get detailed information for a specific segment"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Segment statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as customer_count,
                AVG(c.customer_lifetime_value) as avg_clv,
                AVG(c.churn_risk_score) as avg_churn_risk,
                AVG(cs.health_score) as avg_health_score
            FROM customer_segments cs
            JOIN customers c ON cs.customer_id = c.customer_id
            WHERE cs.segment_name = ?
        ''', (segment_name,))
        
        stats = cursor.fetchone()
        
        if not stats or stats[0] == 0:
            conn.close()
            return jsonify({'error': 'Segment not found'}), 404
        
        # Get customers in segment
        cursor.execute('''
            SELECT 
                c.*,
                cs.health_score,
                cs.engagement_score,
                cs.value_score,
                cs.retention_score
            FROM customers c
            JOIN customer_segments cs ON c.customer_id = cs.customer_id
            WHERE cs.segment_name = ?
        ''', (segment_name,))
        
        customers = [dict_from_row(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'segment_name': segment_name,
            'customer_count': stats[0],
            'avg_clv': round(stats[1] or 0, 2),
            'avg_churn_risk': round(stats[2] or 0, 2),
            'avg_health_score': round(stats[3] or 0, 2),
            'customers': customers
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/churn', methods=['GET'])
def get_churn_predictions():
    """Get churn predictions and high-risk customers"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # High-risk customers
        limit = request.args.get('limit', 50, type=int)
        cursor.execute('''
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
            LIMIT ?
        ''', (limit,))
        
        high_risk = [dict_from_row(row) for row in cursor.fetchall()]
        
        # Risk distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN churn_risk_score > 0.7 THEN 'high_risk'
                    WHEN churn_risk_score > 0.4 THEN 'medium_risk'
                    ELSE 'low_risk'
                END as risk_level,
                COUNT(*) as count,
                SUM(customer_lifetime_value) as total_clv
            FROM customers
            GROUP BY risk_level
        ''')
        
        risk_dist = {row[0]: {'count': row[1], 'total_clv': row[2]} for row in cursor.fetchall()}
        
        conn.close()
        
        return jsonify({
            'high_risk_customers': high_risk,
            'risk_distribution': risk_dist
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/growth', methods=['GET'])
def get_growth_predictions():
    """Get CLV growth predictions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        limit = request.args.get('limit', 50, type=int)
        
        cursor.execute('''
            SELECT 
                c.customer_id,
                c.first_name,
                c.last_name,
                c.customer_lifetime_value,
                cs.segment_name,
                cs.predicted_5yr_clv,
                cs.growth_category,
                (cs.predicted_5yr_clv - c.customer_lifetime_value) as growth_potential
            FROM customers c
            JOIN customer_segments cs ON c.customer_id = cs.customer_id
            WHERE cs.growth_category = 'High Growth' AND c.churn_risk_score < 0.5
            ORDER BY growth_potential DESC
            LIMIT ?
        ''', (limit,))
        
        growth_customers = [dict_from_row(row) for row in cursor.fetchall()]
        
        # Growth distribution
        cursor.execute('''
            SELECT growth_category, COUNT(*) as count
            FROM customer_segments
            GROUP BY growth_category
        ''')
        
        growth_dist = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return jsonify({
            'growth_opportunities': growth_customers,
            'growth_distribution': growth_dist
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/individual', methods=['GET'])
def get_individual_recommendations():
    """Get individual customer recommendations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status', 'pending')
        
        cursor.execute('''
            SELECT 
                r.*,
                c.first_name,
                c.last_name,
                c.email
            FROM recommendations r
            JOIN customers c ON r.customer_id = c.customer_id
            WHERE r.status = ?
            ORDER BY 
                CASE r.action_priority
                    WHEN 'CRITICAL' THEN 1
                    WHEN 'HIGH' THEN 2
                    WHEN 'MEDIUM' THEN 3
                    ELSE 4
                END,
                r.created_at DESC
            LIMIT ?
        ''', (status, limit))
        
        recommendations = [dict_from_row(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'recommendations': recommendations,
            'total_count': len(recommendations)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/customers/<customer_id>', methods=['GET'])
def get_customer_details(customer_id):
    """Get detailed information for a specific customer"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                c.*,
                cf.*,
                cs.*
            FROM customers c
            LEFT JOIN customer_features cf ON c.customer_id = cf.customer_id
            LEFT JOIN customer_segments cs ON c.customer_id = cs.customer_id
            WHERE c.customer_id = ?
        ''', (customer_id,))
        
        customer = cursor.fetchone()
        
        if not customer:
            conn.close()
            return jsonify({'error': 'Customer not found'}), 404
        
        customer_dict = dict_from_row(customer)
        
        # Get recommendations for this customer
        cursor.execute('''
            SELECT * FROM recommendations
            WHERE customer_id = ?
            ORDER BY created_at DESC
        ''', (customer_id,))
        
        customer_dict['recommendations'] = [dict_from_row(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify(customer_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/initialize', methods=['POST'])
def initialize_database():
    """
    Initialize/refresh database from CSV
    POST /api/initialize with CSV file or path
    """
    try:
        csv_path = request.json.get('csv_path', 'customer_data.csv')
        
        # Check if file exists
        if not os.path.exists(csv_path):
            return jsonify({'error': f'CSV file not found: {csv_path}'}), 404
        
        # Run migration
        from database_manager import integrate_database
        db = integrate_database(csv_path)
        db.close()
        
        return jsonify({
            'message': 'Database initialized successfully',
            'database_path': DB_PATH
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/csv', methods=['GET'])
def export_to_csv():
    """Export data back to CSV format"""
    try:
        segment = request.args.get('segment')
        
        conn = get_db_connection()
        
        if segment:
            query = '''
                SELECT c.*, cs.segment_name, cs.health_score
                FROM customers c
                JOIN customer_segments cs ON c.customer_id = cs.customer_id
                WHERE cs.segment_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(segment,))
        else:
            query = '''
                SELECT c.*, cs.segment_name, cs.health_score
                FROM customers c
                LEFT JOIN customer_segments cs ON c.customer_id = cs.customer_id
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        from flask import Response
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment;filename=clv_export_{datetime.now().strftime("%Y%m%d")}.csv'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Check if database exists, if not initialize it
    if not os.path.exists(DB_PATH):
        print("⚠️  Database not found. Run database_manager.py first to initialize.")
        print("    python database_manager.py")
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)