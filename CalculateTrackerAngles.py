import pandas as pd
import matplotlib.pyplot as plt

import TrackerSystemModel as tsm

import pandas as pd
import numpy as np
from datetime import datetime   
import math

import matplotlib.pyplot as plt
from matplotlib import dates

# #region agent log
import json
import os
log_path = r'c:\Users\ColeNoble\Documents\GitHub\geoff-backtracking\.cursor\debug.log'
def log_debug(location, message, data, hypothesis_id):
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":hypothesis_id,"location":location,"message":message,"data":data,"timestamp":os.path.getmtime(__file__) if os.path.exists(__file__) else 0}) + '\n')
    except: pass
# #endregion

dir = 'C:/Users/ColeNoble/Documents/GitHub/geoff-backtracking'

def set_time_index( df, index_name ):
       df.index = pd.to_datetime(df[index_name])
       df = df.drop( [index_name], axis=1)
       return df

# First two arguments are latitude and longitude
# Third and fourth arguments are the width across the tracker and the pitch
# The fifth argument is the maximum phi angle of the tracker
# The sixth argument is the slope of the ground
# The final argument is the time zone for the area. You can get these from the following: import pytz, then just run pytz.all_timezones and it will print out all the ones that you can use
# #region agent log
log_debug("CalculateTrackerAngles.py:26", "Before creating TrackerSystemModelSloped instance", {"tsm_type": str(type(tsm)), "tsm_hasattr": hasattr(tsm, 'TrackerSystemModelSloped')}, "A")
# #endregion
try:
    tracker_model = tsm.TrackerSystemModelSloped(40.26, -83.99, 2.38, 6.69, 52.0, 0, 'America/New_York')
    # #region agent log
    log_debug("CalculateTrackerAngles.py:29", "Successfully created TrackerSystemModelSloped", {"tracker_type": str(type(tracker_model))}, "A")
    # #endregion
except Exception as e:
    # #region agent log
    log_debug("CalculateTrackerAngles.py:31", "Error creating TrackerSystemModelSloped", {"error_type": str(type(e).__name__), "error_msg": str(e)}, "A")
    # #endregion
    raise 

# #region agent log
log_debug("CalculateTrackerAngles.py:35", "Creating date range", {"start": "2025-12-10", "end": "2025-12-11"}, "B")
# #endregion
times = pd.date_range( start = '2025-11-17, end = '2025-11-18', freq = '5min' )
# #region agent log
log_debug("CalculateTrackerAngles.py:37", "Date range created", {"times_count": len(times)}, "B")
# #endregion

angles = []
thetas = []
phis = []

# #region agent log
log_debug("CalculateTrackerAngles.py:44", "Starting loop over times", {"tracker_model_exists": tracker_model is not None}, "C")
# #endregion
for time in times:
    # Theta and phi are the elevation and azimuth respectively. Panel angle is the angle of the panels, but defined with a sign opposite to the standard definition, hence the -1 below. day and back_tracking are just flags.
    # #region agent log
    log_debug("CalculateTrackerAngles.py:47", "Before getTrackerConfiguration call", {"time": str(time), "tracker_model_type": str(type(tracker_model))}, "C")
    # #endregion
    try:
        day, back_tracking, panel_angle, theta, phi, _ = tracker_model.getTrackerConfiguration( time )
        # #region agent log
        log_debug("CalculateTrackerAngles.py:50", "After getTrackerConfiguration call", {"day": day, "back_tracking": back_tracking, "panel_angle": panel_angle, "theta": theta, "phi": phi}, "C")
        # #endregion
    except Exception as e:
        # #region agent log
        log_debug("CalculateTrackerAngles.py:53", "Error in getTrackerConfiguration", {"error_type": str(type(e).__name__), "error_msg": str(e), "time": str(time)}, "C")
        # #endregion
        raise
    if theta > 0:
        angles.append( -1 * panel_angle)   
    else:
        angles.append( 0 )
    thetas.append(theta)
    phis.append(phi)
    
# #region agent log
log_debug("CalculateTrackerAngles.py:63", "Creating DataFrame", {"angles_count": len(angles), "thetas_count": len(thetas), "phis_count": len(phis)}, "D")
# #endregion
df = pd.DataFrame( { 'PanelAngle' : angles,  'Elevation' : thetas, 'Azimuths' : phis }, index = times )

df_theoretical = df * 180.0 / math.pi

# #region agent log
log_debug("CalculateTrackerAngles.py:68", "Before saving to Excel", {"dir": dir, "full_path": os.path.join(dir, 'theoretical_angles.xlsx')}, "E")
# #endregion
try:
    df_theoretical.to_excel(os.path.join(dir, 'theoretical_angles.xlsx'))
    # #region agent log
    log_debug("CalculateTrackerAngles.py:71", "Successfully saved to Excel", {}, "E")
    # #endregion
except Exception as e:
    # #region agent log
    log_debug("CalculateTrackerAngles.py:74", "Error saving to Excel", {"error_type": str(type(e).__name__), "error_msg": str(e)}, "E")
    # #endregion
    raise
