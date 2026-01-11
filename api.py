from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from data_analysis import DataAnalyzer
from predictive_analytics import PredictiveAnalytics
from recommendations import RecommendationEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_data_for_json(obj):
    """
    Recursively clean data to make it JSON-compliant.
    Replaces NaN, Infinity with None or 0.
    """
    if isinstance(obj, dict):
        return {k: clean_data_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None  # or 0, depending on your preference
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif pd.isna(obj):
        return None
    return obj

@app.get("/dashboard")
def get_dashboard_data():
    try:
        analyzer = DataAnalyzer("customer_data.csv")
        
        predictor = PredictiveAnalytics(analyzer.df)
        predictor.customer_segmentation()
        predictor.churn_prediction_model()
        predictor.lifetime_value_prediction()
        predictor.generate_customer_scores()
        
        recommender = RecommendationEngine(predictor.df)
        segment_strategies = recommender.generate_segment_strategies()
        individual_recs = recommender.generate_individual_recommendations()
        roi = recommender.calculate_roi_projections()

        # Convert DataFrame to dict and clean
        customers_data = predictor.df.to_dict(orient="records")
        customers_data = clean_data_for_json(customers_data)
        
        response = {
            "customers": customers_data,
            "segment_strategies": clean_data_for_json(segment_strategies),
            "individual_recommendations": clean_data_for_json(individual_recs),
            "roi": clean_data_for_json(roi),
            "summary": {
                "total_customers": int(len(predictor.df)),
                "total_clv": float(predictor.df["customer_lifetime_value"].sum()),
                "churn_rate": float(predictor.df["is_churned"].mean())
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in /dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "customers": [],
            "segment_strategies": {},
            "individual_recommendations": [],
            "roi": {},
            "summary": {
                "total_customers": 0,
                "total_clv": 0,
                "churn_rate": 0
            }
        }

@app.get("/")
def read_root():
    return {"message": "CLV Analytics API is running", "endpoints": ["/dashboard"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}