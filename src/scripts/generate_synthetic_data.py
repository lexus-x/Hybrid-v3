"""Generate synthetic radar and AIS tracking data for testing.

Creates realistic vessel movements near Busan port to replace the confidential dataset.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def main():
    np.random.seed(42)
    
    # Target directory
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "busan")
    os.makedirs(raw_dir, exist_ok=True)
    
    start_time = datetime(2025, 8, 1, 15, 0, 0)
    n_vessels = 80
    n_points = 25
    dt_step = 30.0  # seconds
    
    # AIS systematic bias (e.g. 22m East, 10m North)
    bias_east = 22.0
    bias_north = 10.0
    
    ais_rows = []
    radar_rows = []
    
    for i in range(n_vessels):
        mmsi = str(440000000 + i)
        target_id = f"T_{i:03d}"
        
        # Initial positions near Busan port
        lat = 35.08 + np.random.uniform(-0.04, 0.04)
        lon = 129.04 + np.random.uniform(-0.04, 0.04)
        
        # Constant heading (COG) and speed (SOG)
        cog = np.random.uniform(0.0, 360.0)
        sog = np.random.uniform(5.0, 25.0)  # knots
        
        # Convert speed to meters per second
        speed_mps = sog * 0.514444
        
        t_current = start_time
        
        for p in range(n_points):
            t_sec = p * dt_step
            # Time offset
            t_current = start_time + timedelta(seconds=t_sec)
            time_str = f'="{t_current.strftime("%Y-%m-%d %H:%M:%S.%f")}"'
            
            # Update position (straight line)
            rad_cog = np.radians(cog)
            dist_m = speed_mps * dt_step
            
            d_lat = (dist_m * np.cos(rad_cog)) / 110540.0
            d_lon = (dist_m * np.sin(rad_cog)) / (111320.0 * np.cos(np.radians(lat)))
            
            lat += d_lat
            lon += d_lon
            
            # Slight noise in heading and speed
            cog_noise = np.random.normal(0.0, 1.0)
            sog_noise = np.random.normal(0.0, 0.5)
            
            point_cog = (cog + cog_noise) % 360.0
            point_sog = max(0.0, sog + sog_noise)
            
            # Radar point (no systematic bias, small random GPS noise)
            r_lat = lat + np.random.normal(0.0, 5.0) / 110540.0
            r_lon = lon + np.random.normal(0.0, 5.0) / (111320.0 * np.cos(np.radians(lat)))
            
            radar_rows.append({
                "targetId": target_id,
                "longitude": f"{r_lon:.7f}",
                "latitude": f"{r_lat:.7f}",
                "cog": f"{point_cog:.1f}",
                "sog": f"{point_sog:.1f}",
                "dateTime": time_str
            })
            
            # AIS point (with systematic bias and small random GPS noise)
            # Add bias
            b_lat = lat + bias_north / 110540.0
            b_lon = lon + bias_east / (111320.0 * np.cos(np.radians(lat)))
            
            # Add GPS noise
            a_lat = b_lat + np.random.normal(0.0, 5.0) / 110540.0
            a_lon = b_lon + np.random.normal(0.0, 5.0) / (111320.0 * np.cos(np.radians(lat)))
            
            ais_data = {
                "cog": point_cog,
                "sog": point_sog
            }
            
            ais_rows.append({
                "mmsi": mmsi,
                "message_type": "1",
                "longitude": f"{a_lon:.7f}",
                "latitude": f"{a_lat:.7f}",
                "date_time": time_str,
                "data": json.dumps(ais_data)
            })
            
    # Save datasets
    df_ais = pd.DataFrame(ais_rows)
    df_ais.to_csv(os.path.join(raw_dir, "ais_synthetic.csv"), index=False)
    
    df_radar = pd.DataFrame(radar_rows)
    df_radar.to_csv(os.path.join(raw_dir, "radartarget_synthetic.csv"), index=False)
    
    print(f"Generated {len(ais_rows)} AIS points and {len(radar_rows)} radar points successfully.")

if __name__ == "__main__":
    main()
