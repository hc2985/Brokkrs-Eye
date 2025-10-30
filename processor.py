import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model import Model_A

def process_imu_data(input_csv_path, output_csv_path=None, progress_callback=None):
    """
    Process IMU data from CSV and generate yaw rate classifications.

    Args:
        input_csv_path: Path to input CSV file (same format as data.csv)
        output_csv_path: Path for output CSV (if None, auto-generates filename)
        progress_callback: Optional callback function(message, progress_percent)

    Returns:
        output_csv_path: Path to the generated output file
    """

    def update_progress(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)
        print(msg)

    try:
        # Configuration
        window_size = 100
        stride = 10
        weights_filename = "Model_A_B500_E300_V2.hdf5"

        # Auto-generate output filename if not provided
        if output_csv_path is None:
            base_dir = os.path.dirname(input_csv_path)
            base_name = os.path.basename(input_csv_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_csv_path = os.path.join(base_dir, f"{name_without_ext}_processed.csv")

        # Load model
        update_progress("Loading model...", 10)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, weights_filename)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        model = Model_A(window_size)
        model.load_weights(weights_path, by_name=True, skip_mismatch=False)
        update_progress("Model loaded successfully", 20)

        # Load and preprocess data
        update_progress("Loading input data...", 25)
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input file not found: {input_csv_path}")

        df = pd.read_csv(input_csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Validate required columns
        required_cols = ['timestamp', 'milliseconds', 'accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        update_progress("Preprocessing data...", 30)

        # Build absolute timestamps
        df["abs_time"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S") \
                           + pd.to_timedelta(df["milliseconds"], unit="ms")
        timestamps = (df["abs_time"] - df["abs_time"].iloc[0]).dt.total_seconds().to_numpy()

        # Raw sensors
        acc_raw = df[["accx", "accy", "accz"]].to_numpy().astype(np.float32)
        gyro_raw = df[["gyrox", "gyroy", "gyroz"]].to_numpy().astype(np.float32)

        # Frame remapping (Bike → Model frame)
        update_progress("Remapping sensor frames...", 35)
        acc = np.column_stack([acc_raw[:, 2], -acc_raw[:, 1], acc_raw[:, 0]])
        gyro = np.column_stack([gyro_raw[:, 2], -gyro_raw[:, 1], -gyro_raw[:, 0]])

        # Normalize accelerometer
        norm = np.linalg.norm(acc, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        acc_normalized = acc / norm

        # Gyro to rad/s
        gyro_rad = np.deg2rad(gyro)

        # Create overlapping windows
        update_progress("Creating data windows...", 40)
        N = len(acc_normalized)
        idxs = list(range(0, N - window_size + 1, stride))

        def stack_windows(arr, starts, w):
            return np.stack([arr[s:s + w] for s in starts], axis=0)

        acc_windows = stack_windows(acc_normalized, idxs, window_size)
        gyro_windows = stack_windows(gyro_rad, idxs, window_size)

        # Per-window sampling frequency
        win_durations = np.array([
            max(timestamps[s + window_size - 1] - timestamps[s], 1e-6) for s in idxs
        ], dtype=np.float64)
        fs_per_window = (window_size - 1) / win_durations
        fs_input = fs_per_window.reshape(-1, 1).astype(np.float32)

        times = np.array([timestamps[s + window_size - 1] for s in idxs], dtype=np.float64)

        # Run model inference
        update_progress("Running model inference...", 50)
        pred_quats = model.predict([acc_windows, gyro_windows, fs_input], verbose=0)

        # Quaternion to Euler
        update_progress("Computing orientation...", 70)
        def quat_to_euler(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - x*z), -1.0, 1.0))
            yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            return roll, pitch, yaw

        roll_pred, pitch_pred, yaw_pred = quat_to_euler(pred_quats)

        # Calibration (bias removal)
        CALIBRATION_SAMPLES = min(10, len(roll_pred))
        roll_offset  = np.mean(roll_pred[:CALIBRATION_SAMPLES])
        pitch_offset = np.mean(pitch_pred[:CALIBRATION_SAMPLES])
        roll_pred  = roll_pred  - roll_offset
        pitch_pred = pitch_pred - pitch_offset

        # Integrate gyro for yaw
        update_progress("Integrating yaw from gyroscope...", 75)
        num_windows = len(idxs)
        yaw_integrated = np.zeros(num_windows, dtype=np.float64)
        end_indices = np.array([s + window_size - 1 for s in idxs], dtype=int)

        for i in range(1, num_windows):
            prev_end = end_indices[i - 1]
            curr_end = end_indices[i]
            start = prev_end + 1
            end = curr_end + 1

            if start >= end or end > len(gyro_rad):
                dt_step = max(times[i] - times[i - 1], 1e-6)
                gyro_z_avg = gyro_rad[min(curr_end, len(gyro_rad) - 1), 2]
            else:
                dt_step = timestamps[end - 1] - timestamps[start - 1]
                dt_step = max(dt_step, 1e-9)
                gyro_z_avg = np.mean(gyro_rad[start:end, 2])

            yaw_integrated[i] = yaw_integrated[i - 1] + gyro_z_avg * dt_step

        # Yaw rate classification
        update_progress("Classifying turn types...", 85)
        yaw_deg_int = np.degrees(yaw_integrated)
        yaw_deg_int_unwrapped = np.degrees(np.unwrap(np.radians(yaw_deg_int), discont=np.pi))

        yaw_deltas = np.diff(yaw_deg_int_unwrapped)
        dt_steps = np.diff(times)
        dt_steps = np.where(dt_steps <= 0, 1e-6, dt_steps)
        yaw_rate = yaw_deltas / dt_steps
        yaw_times = times[1:]

        def classify(rate):
            mag = abs(rate)
            if mag < 2.0:
                return "straight"
            elif mag < 5.0:
                return "slight right" if rate > 0 else "slight left"
            elif mag < 15.0:
                return "normal right" if rate > 0 else "normal left"
            else:
                return "major right" if rate > 0 else "major left"

        labels = [classify(r) for r in yaw_rate]

        # Create output DataFrame with yaw rate data
        yaw_df = pd.DataFrame({
            "time(s)": yaw_times,
            "delta_yaw(deg)": yaw_deltas,
            "dt(s)": dt_steps,
            "yaw_rate(deg/s)": yaw_rate,
            "turn_type": labels
        })

        # Merge with original data based on timestamp alignment
        # Map yaw_times back to original dataframe indices
        update_progress("Merging results with original data...", 90)

        # Create a mapping from window end times to yaw rate data
        # We'll add the yaw rate columns to the original dataframe
        # For rows that correspond to window end times

        # Initialize new columns in original dataframe
        df["time(s)"] = timestamps
        df["delta_yaw(deg)"] = np.nan
        df["dt(s)"] = np.nan
        df["yaw_rate(deg/s)"] = np.nan
        df["turn_type"] = ""

        # Map yaw rate data to closest timestamps in original data
        for i, t in enumerate(yaw_times):
            # Find closest timestamp in original data
            idx = np.argmin(np.abs(timestamps - t))
            df.loc[idx, "delta_yaw(deg)"] = yaw_deltas[i]
            df.loc[idx, "dt(s)"] = dt_steps[i]
            df.loc[idx, "yaw_rate(deg/s)"] = yaw_rate[i]
            df.loc[idx, "turn_type"] = labels[i]

        # Save output
        update_progress("Saving output file...", 95)
        df.to_csv(output_csv_path, index=False, float_format="%.6f")

        update_progress(f"Processing complete! Output saved to: {output_csv_path}", 100)

        return output_csv_path

    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        update_progress(error_msg, None)
        raise Exception(error_msg)


if __name__ == "__main__":
    # Test the processor
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        process_imu_data(input_file, output_file)
    else:
        print("Usage: python processor.py <input_csv> [output_csv]")
