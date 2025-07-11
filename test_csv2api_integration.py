#!/usr/bin/env python3
"""
Quick integration test for csv2api
"""
import sys
import os
sys.path.append('csv2api')

def test_csv2api():
    print("=== CSV2API Integration Test ===")
    
    # Test 1: Import check
    try:
        import app
        print("✅ Successfully imported csv2api app")
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False
    
    # Test 2: Check if Flask app exists
    try:
        if hasattr(app, 'app'):
            print("✅ Flask app object found")
        else:
            print("❌ Flask app object not found")
    except Exception as e:
        print(f"❌ Error checking Flask app: {e}")
    
    # Test 3: Test CSV file processing
    if os.path.exists('test_transactions.csv'):
        print("✅ Test CSV file available")
        
        # Try to process it
        try:
            import pandas as pd
            df = pd.read_csv('test_transactions.csv')
            print(f"✅ Successfully loaded CSV with {len(df)} rows")
            print("   Columns:", list(df.columns))
        except Exception as e:
            print(f"❌ Error processing CSV: {e}")
    else:
        print("❌ Test CSV file not found")
    
    return True

if __name__ == "__main__":
    test_csv2api()
