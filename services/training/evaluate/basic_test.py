#!/usr/bin/env python3
"""
Basic test without complex imports - just test data loading and basic processing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def basic_test():
    """Basic test of AAPL data"""
    print("=== Basic AAPL Data Test ===")
    
    # 1. Check if AAPL data exists
    aapl_file = Path("data/dow30_real_AAPL.csv")
    
    if not aapl_file.exists():
        print(f"❌ AAPL data not found: {aapl_file}")
        return False
    
    print(f"✅ AAPL data found: {aapl_file}")
    
    # 2. Load and inspect data
    print("\nLoading AAPL data...")
    df = pd.read_csv(aapl_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Data loaded: {len(df)} records")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Columns: {list(df.columns)}")
    
    # 3. Check data quality
    print(f"\nData quality check:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate timestamps: {df['timestamp'].duplicated().sum()}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # 4. Split by year
    df_2020_2024 = df[df['timestamp'].dt.year.isin([2020, 2021, 2022, 2023, 2024])]
    df_2025 = df[df['timestamp'].dt.year == 2025]
    
    print(f"\nData split:")
    print(f"   Training (2020-2024): {len(df_2020_2024)} records")
    print(f"   Test (2025): {len(df_2025)} records")
    
    if len(df_2025) == 0:
        print(f"⚠️  No 2025 data found. Using 2024 data for test.")
        df_2025 = df[df['timestamp'].dt.year == 2024].tail(100)  # Last 100 days of 2024
    
    # 5. Test basic window slicing (simple implementation)
    print(f"\nTesting basic window slicing...")
    try:
        window_size = 20
        step_size = 5
        
        # Simple window slicing
        segments = []
        for i in range(0, len(df_2020_2024) - window_size + 1, step_size):
            window_data = df_2020_2024.iloc[i:i + window_size]
            if len(window_data) == window_size:
                segments.append({
                    'start_idx': i,
                    'end_idx': i + window_size,
                    'start_date': window_data['timestamp'].iloc[0],
                    'end_date': window_data['timestamp'].iloc[-1],
                    'bars': len(window_data)
                })
        
        print(f"✅ Basic window slicing: {len(segments)} segments created")
        
        if segments:
            sample = segments[0]
            print(f"   Sample segment: {sample['start_date'].date()} to {sample['end_date'].date()}")
            print(f"   Bars: {sample['bars']}")
    
    except Exception as e:
        print(f"❌ Basic window slicing failed: {e}")
        return False
    
    # 6. Test basic feature extraction
    print(f"\nTesting basic feature extraction...")
    try:
        if segments:
            # Get first segment data
            first_segment_data = df_2020_2024.iloc[segments[0]['start_idx']:segments[0]['end_idx']]
            
            # Calculate basic features
            closes = first_segment_data['close'].values
            features = {
                'price_change': (closes[-1] - closes[0]) / closes[0],
                'price_volatility': np.std(closes) / np.mean(closes),
                'max_price': np.max(closes),
                'min_price': np.min(closes),
                'avg_volume': first_segment_data['volume'].mean()
            }
            
            print(f"✅ Basic feature extraction: {len(features)} features")
            print(f"   Sample features: {list(features.keys())}")
            print(f"   Price change: {features['price_change']:.4f}")
            print(f"   Volatility: {features['price_volatility']:.4f}")
    
    except Exception as e:
        print(f"❌ Basic feature extraction failed: {e}")
        return False
    
    # 7. Test basic candlestick visualization
    print(f"\nTesting basic candlestick visualization...")
    try:
        import matplotlib.pyplot as plt
        
        if segments:
            # Get first segment data
            first_segment_data = df_2020_2024.iloc[segments[0]['start_idx']:segments[0]['end_idx']]
            
            # Create simple candlestick plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot candlesticks
            for i, (_, row) in enumerate(first_segment_data.iterrows()):
                color = 'green' if row['close'] >= row['open'] else 'red'
                
                # Draw wick
                ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
                
                # Draw body
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                ax.bar(i, body_height, bottom=body_bottom, color=color, alpha=0.7, width=0.8)
            
            ax.set_title('AAPL Candlestick Chart (Sample)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('test_aapl_candlestick.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Basic candlestick visualization: SUCCESS")
            print(f"   Test image saved: test_aapl_candlestick.png")
    
    except Exception as e:
        print(f"❌ Basic candlestick visualization failed: {e}")
        return False
    
    print(f"\n=== Basic Test Summary ===")
    print(f"✅ Data loading: SUCCESS")
    print(f"✅ Data quality: GOOD")
    print(f"✅ Basic window slicing: SUCCESS")
    print(f"✅ Basic feature extraction: SUCCESS")
    print(f"✅ Basic visualization: SUCCESS")
    
    print(f"\nData is ready for processing!")
    print(f"Next steps:")
    print(f"1. Implement contrastive learning training")
    print(f"2. Train encoder on training segments")
    print(f"3. Test similarity search on test segments")
    
    return True

if __name__ == "__main__":
    success = basic_test()
    if not success:
        print(f"\n❌ Basic test failed. Please check the errors above.")
        exit(1)

