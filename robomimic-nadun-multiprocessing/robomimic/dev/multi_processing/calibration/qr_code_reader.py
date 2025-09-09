import pickle
import cv2
import numpy as np
import time

def extract_qr_timestamps(data_path):
    """
    Read pickle file containing camera data and extract QR code timestamps
    using OpenCV's QR code detector
    
    Args:
        data_path: Path to the pickle file containing camera data
    """
    # Initialize QR Code detector
    qr_detector = cv2.QRCodeDetector()
    cv2.namedWindow("qr_code_reader")
    
    # Load the pickle file
    with open(data_path, "rb") as f:
        data_list = pickle.load(f)
    
    # Process each frame
    for data in data_list:
        rgb = data['rgb']
        t_cap = data['camera_capture_timestamp']
        t_recv = data['camera_receive_timestamp']
        t_cal = data['timestamp']
        print(f"t_cap {t_cap}")

        cv2.imshow("qr_code_reader", rgb)
        cv2.waitKey(1)
        
        # Decode QR code from RGB image
        # OpenCV expects BGR format
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = rgb
            
        retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(bgr)
        
        # Extract timestamp if QR code is found
        data['QR_code_time'] = None
        if retval:
            for qr_data in decoded_info:
                if qr_data:  # Check if data is not empty
                    try:
                        qr_timestamp = float(qr_data)
                        
                        # Add QR code time to the data dictionary
                        data['QR_code_time'] = qr_timestamp
                        
                        # Optional: Print timing information
                        print(f"Frame timing:")
                        print(f"  QR timestamp: {qr_timestamp}")
                        print(f"  Capture time: {t_cap}")
                        print(f"  Receive time: {t_recv}")
                        print(f"  Calculated time: {t_cal}")
                        print(f"  Latency (receive - QR): {t_recv - qr_timestamp:.3f}s")
                        print("---")
                        
                    except ValueError as e:
                        print(f"Error parsing QR code timestamp: {e}")
    
    return data_list

if __name__ == "__main__":
    # Path to your pickle file
    # data_path = "kinect_data_list.pkl"
    data_path = "../envs/zed_data_list60.pkl"
    # Process the data
    processed_data = extract_qr_timestamps(data_path)

    # # Save processed data back to pickle file
    # output_path = data_path.replace('.pkl', '_with_qr.pkl')
    # with open(output_path, 'wb') as f:
    #     pickle.dump(processed_data, f)
    
    # Print summary statistics
    total_frames = len(processed_data)
    frames_with_qr = sum(1 for data in processed_data if data['QR_code_time'] is not None)
    
    print("\nSummary:")
    print(f"Total frames: {total_frames}")
    print(f"Frames with QR code: {frames_with_qr}")
    print(f"QR code detection rate: {frames_with_qr/total_frames*100:.1f}%")
    
    # Optional: Calculate timing statistics for frames with QR codes
    if frames_with_qr > 0:
        latencies = []
        for data in processed_data:
            if data['QR_code_time'] is not None:
                qr_time = data['QR_code_time']
                latency = data['camera_receive_timestamp'] - qr_time
                latencies.append(latency)
        
        latencies = np.array(latencies)
        print("\nLatency statistics (receive time - QR time):")
        print(f"  Mean: {np.mean(latencies)*1000:.1f}ms")
        print(f"  Std: {np.std(latencies)*1000:.1f}ms")
        print(f"  Min: {np.min(latencies)*1000:.1f}ms")
        print(f"  Max: {np.max(latencies)*1000:.1f}ms")