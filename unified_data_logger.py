#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified NPZ Data Logger for Soft Robot System
FIXED VERSION with proper FFmpeg error handling
"""

import numpy as np
import threading
from datetime import datetime, timezone
import os
import json
import pytz
from collections import defaultdict

class UnifiedDataLogger:
    """
    Optimized logger with separate files for sensors.
    Uses ABSOLUTE PST timestamps (Unix epoch + timezone info)
    
    Creates:
    - {session}_sensors.npz: IMU, pressure, motor, joystick data
    """
    
    def __init__(self, log_dir="robot_logs", session_name=None, debug=True):
        """
        Args:
            log_dir: Directory to save log files
            session_name: Optional custom session name (auto-generated if None)
            debug: Print detailed logging info
        """
        self.log_dir = log_dir
        self.timezone = pytz.timezone('America/Los_Angeles')  # PST/PDT
        self.debug = debug

        # Session info
        self.session_start = datetime.now(self.timezone)
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        timestamp_dir=os.path.join(log_dir,self.session_id)
        self.log_dir=os.path.join(timestamp_dir, subsystem)

        os.makedirs(self.log_dir,exist_ok=True)

        self.session_name = session_name
        
        # File paths
        self.sensors_npz_path = os.path.join(log_dir, f"{session_name}_sensors.npz")
        self.meta_filepath = os.path.join(log_dir, f"{session_name}_meta.json")
        
        # Data buffers
        self.data = defaultdict(list)  # Sensor data
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.sample_counts = defaultdict(int) 
        
        print(f"[UnifiedLogger] Session: {session_name}")
        print(f"[UnifiedLogger] Timezone: {self.timezone}")
        print(f"[UnifiedLogger] Session start: {self.session_start.isoformat()}")
        print(f"[UnifiedLogger] Sensors NPZ: {os.path.abspath(self.sensors_npz_path)}")
        print(f"[UnifiedLogger] Metadata: {os.path.abspath(self.meta_filepath)}")
    
    def _get_timestamp(self):
        """
        Get current timestamp as Unix epoch (float seconds since 1970-01-01 UTC).
        Includes microsecond precision.
        """
        return datetime.now(self.timezone).timestamp()
    
    # ========== SENSOR LOGGING METHODS ==========
    
    def log_imu(self, imu_id, qw, qx, qy, qz, roll, pitch, yaw, encoder1, encoder2):
        """Log IMU quaternion, Euler angles, and encoder data."""
        with self.lock:
            self.data['timestamp'].append(self._get_timestamp())
            self.data['data_type'].append(0)  # 0 = IMU data
            self.data['imu_id'].append(imu_id)
            self.data['qw'].append(qw)
            self.data['qx'].append(qx)
            self.data['qy'].append(qy)
            self.data['qz'].append(qz)
            self.data['roll'].append(roll)
            self.data['pitch'].append(pitch)
            self.data['yaw'].append(yaw)
            self.data['encoder1'].append(encoder1)
            self.data['encoder2'].append(encoder2)
            self.data['pressure'].append(np.nan)
            self.data['motor_velocity'].append(np.nan)
            self.data['torque_data'].append(np.nan)
            self.data['motor_pos_data'].append(np.nan)
            self.data['ax'].append(np.nan)
            self.data['ay'].append(np.nan)
            self.data['az'].append(np.nan)
            self.data['gx'].append(np.nan)
            self.data['gy'].append(np.nan)
            self.data['gz'].append(np.nan)
            self.data['mx'].append(np.nan)
            self.data['my'].append(np.nan)
            self.data['mz'].append(np.nan)
            
            self.sample_counts['imu'] += 1
    
    def log_imu_accel_gyro_magno(self, imu_id,ax,ay,az,gx,gy,gz,mx,my,mz):
        """Log IMU quaternion, Euler angles, and encoder data."""
        with self.lock:
            self.data['timestamp'].append(self._get_timestamp())
            self.data['data_type'].append(3)  # 3 = IMU data in accel gyro magno
            self.data['imu_id'].append(imu_id)
            self.data['ax'].append(ax)
            self.data['ay'].append(ay)
            self.data['az'].append(az)
            self.data['gx'].append(gx)
            self.data['gy'].append(gy)
            self.data['gz'].append(gz)
            self.data['mx'].append(mx)
            self.data['my'].append(my)
            self.data['mz'].append(mz)
            self.data['pressure'].append(np.nan)
            self.data['motor_velocity'].append(np.nan)
            self.data['torque_data'].append(np.nan)
            self.data['motor_pos_data'].append(np.nan)
            self.data['qw'].append(np.nan)
            self.data['qx'].append(np.nan)
            self.data['qy'].append(np.nan)
            self.data['qz'].append(np.nan)
            self.data['roll'].append(np.nan)
            self.data['pitch'].append(np.nan)
            self.data['yaw'].append(np.nan)
            self.data['encoder1'].append(np.nan)
            self.data['encoder2'].append(np.nan)
            
            self.sample_counts['imu_accel_gyro_magno'] += 1
    
    def log_pressure(self, pressure):
        """Log pressure sensor reading."""
        with self.lock:
            self.data['timestamp'].append(self._get_timestamp())
            self.data['data_type'].append(1)  # 1 = Pressure data
            self.data['imu_id'].append(-1)
            self.data['qw'].append(np.nan)
            self.data['qx'].append(np.nan)
            self.data['qy'].append(np.nan)
            self.data['qz'].append(np.nan)
            self.data['roll'].append(np.nan)
            self.data['pitch'].append(np.nan)
            self.data['yaw'].append(np.nan)
            self.data['encoder1'].append(np.nan)
            self.data['encoder2'].append(np.nan)
            self.data['pressure'].append(pressure)
            self.data['motor_velocity'].append(np.nan)
            self.data['torque_data'].append(np.nan)
            self.data['motor_pos_data'].append(np.nan)
            self.data['ax'].append(np.nan)
            self.data['ay'].append(np.nan)
            self.data['az'].append(np.nan)
            self.data['gx'].append(np.nan)
            self.data['gy'].append(np.nan)
            self.data['gz'].append(np.nan)
            self.data['mx'].append(np.nan)
            self.data['my'].append(np.nan)
            self.data['mz'].append(np.nan)
            
            self.sample_counts['pressure'] += 1
    
    def log_motor(self, velocity, torque_data=None, motor_pos_data=None):
        """Log motor velocity and IQ current."""
        with self.lock:
            self.data['timestamp'].append(self._get_timestamp())
            self.data['data_type'].append(2)  # 2 = Motor data
            self.data['imu_id'].append(-1)
            self.data['qw'].append(np.nan)
            self.data['qx'].append(np.nan)
            self.data['qy'].append(np.nan)
            self.data['qz'].append(np.nan)
            self.data['roll'].append(np.nan)
            self.data['pitch'].append(np.nan)
            self.data['yaw'].append(np.nan)
            self.data['encoder1'].append(np.nan)
            self.data['encoder2'].append(np.nan)
            self.data['pressure'].append(np.nan)
            self.data['motor_velocity'].append(velocity)
            self.data['torque_data'].append(torque_data if torque_data is not None else np.nan)
            self.data['motor_pos_data'].append(motor_pos_data if motor_pos_data is not None else np.nan)
            self.data['ax'].append(np.nan)
            self.data['ay'].append(np.nan)
            self.data['az'].append(np.nan)
            self.data['gx'].append(np.nan)
            self.data['gy'].append(np.nan)
            self.data['gz'].append(np.nan)
            self.data['mx'].append(np.nan)
            self.data['my'].append(np.nan)
            self.data['mz'].append(np.nan)
            
            self.sample_counts['motor'] += 1
    
    def log_joystick(self, axes, buttons):
        """Log joystick state (axes and buttons)."""
        with self.lock:
            if 'joy_timestamp' not in self.data:
                self.data['joy_timestamp'] = []
                self.data['joy_axes'] = []
                self.data['joy_buttons'] = []
            
            self.data['joy_timestamp'].append(self._get_timestamp())
            self.data['joy_axes'].append(np.array(axes))
            self.data['joy_buttons'].append(np.array(buttons))
            
            self.sample_counts['joystick'] += 1
    
    # ========== SAVE METHODS ==========
    
    def save(self):
        """Save all buffered data to files."""
        session_end = datetime.now(self.timezone)
        duration = (session_end - self.session_start).total_seconds()
        
        # ===== SAVE SENSOR DATA =====
        with self.lock:
            arrays_to_save = {}
            
            # Main sensor data arrays
            for key in ['timestamp', 'data_type', 'imu_id', 'qw', 'qx', 'qy', 'qz',
                       'roll', 'pitch', 'yaw', 'encoder1', 'encoder2', 
                       'pressure', 'motor_velocity', 'torque_data', 'motor_pos_data','ax','ay','az','gx','gy','gz','mx','my','mz']:
                if key in self.data and len(self.data[key]) > 0:
                    arrays_to_save[key] = np.array(self.data[key])
            
            # Joystick data (object arrays)
            if 'joy_timestamp' in self.data and len(self.data['joy_timestamp']) > 0:
                arrays_to_save['joy_timestamp'] = np.array(self.data['joy_timestamp'])
                arrays_to_save['joy_axes'] = np.array(self.data['joy_axes'], dtype=object)
                arrays_to_save['joy_buttons'] = np.array(self.data['joy_buttons'], dtype=object)
            
            #Save sensors NPZ
            if arrays_to_save:
                np.savez_compressed(self.sensors_npz_path, **arrays_to_save)
                sensor_samples = len(self.data.get('timestamp', []))
                sensor_size_mb = os.path.getsize(self.sensors_npz_path) / (1024 * 1024)
                print(f"[UnifiedLogger] Saved {sensor_samples} sensor samples to sensors.npz ({sensor_size_mb:.2f} MB)")
        
        # ===== CREATE METADATA =====
        metadata = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'end_time': session_end.isoformat(),
            'duration_seconds': duration,
            'timezone': str(self.timezone),
            'sample_counts': dict(self.sample_counts),
            'total_samples': sum(self.sample_counts.values()),
            
            'timestamp_format': {
                'type': 'absolute',
                'unit': 'seconds since Unix epoch (1970-01-01 00:00:00 UTC)',
                'precision': 'microseconds',
                'timezone': 'America/Los_Angeles (PST/PDT)',
            },
            
            'files': {
                'sensors': os.path.basename(self.sensors_npz_path),
            },

            
            'data_types': {
                '0': 'IMU (quaternion, euler, encoders)',
                '1': 'Pressure',
                '2': 'Motor velocity + IQ current'
            },
            
            'file_format': 'Sensors: NumPy NPZ (compressed)',
        }
        
        # Calculate rates
        if duration > 0:
            for key, count in self.sample_counts.items():
                metadata[f'{key}_rate_hz'] = count / duration
        if getattr(self, "context", None):
            metadata.update(self.context)
        
        if getattr(self, "context", None):
            metadata.update(self.context)
        
        # Save metadata
        with open(self.meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[UnifiedLogger] Metadata saved to {self.meta_filepath}")
    
    def close(self):
        """Save data and close the logger."""
        self.save()
        
        elapsed = (datetime.now(self.timezone) - self.session_start).total_seconds()
        print(f"\n[UnifiedLogger] ========== SESSION SUMMARY ==========")
        print(f"  Duration: {elapsed:.2f} seconds")
        print(f"  Camera logging ended : {datetime.now(self.timezone)}")
        print(f"  Sample counts:")
        for data_type, count in sorted(self.sample_counts.items()):
            rate = count / elapsed if elapsed > 0 else 0
            print(f"    {data_type:12s}: {count:6d} samples ({rate:6.2f} Hz)")
        
        print(f"  Files saved:")
        if os.path.exists(self.sensors_npz_path):
            size_mb = os.path.getsize(self.sensors_npz_path) / (1024 * 1024)
            print(f"    Sensors: {self.sensors_npz_path} ({size_mb:.2f} MB)")
        

# ========== HELPER FUNCTIONS ==========

def timestamp_to_datetime(timestamp, timezone_str='America/Los_Angeles'):
    """Convert Unix timestamp to datetime object in specified timezone."""
    tz = pytz.timezone(timezone_str)
    return datetime.fromtimestamp(timestamp, tz=tz)