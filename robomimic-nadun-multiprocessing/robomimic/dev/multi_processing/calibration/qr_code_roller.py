import qrcode
import time
import cv2
import numpy as np
import gc

from line_profiler_pycharm import profile

from robomimic.utils.time_utils import precise_wait, precise_sleep

"""
For Kinect, on RL2-WS4, measured on Dec 24, 2024 by CZY
QR code latency: 0.002148578832815359 @ 60Hz+, for 10Hz, it is 0.0177
Haven't figured out why there is a difference related to th
"""
@profile
def generate_qr_with_timestamp():

    refresh_rate = 30.0

    QR_code_latency = []
    display_latency = []
    while True:
        # Get current timestamp
        # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        current_time = round(time.time(), 5)
        
        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=30, border=5)
        qr.add_data(current_time)
        # qr.make(fit=True) # add more time...
        
        # Convert to image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to numpy array for OpenCV
        qr_numpy = np.array(qr_image)
        qr_numpy = (qr_numpy * 255).astype(np.uint8)

        # Display QR
        qr_time = time.time()
        cv2.imshow('QR Code', qr_numpy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        display_time = time.time()

        QR_code_latency.append(qr_time - current_time)
        display_latency.append(display_time-qr_time)
        # print(f"QR code display time: {display_time - current_time}")

        precise_sleep(1/refresh_rate - (time.time()-current_time))

    cv2.destroyAllWindows()
    print(f"qr latency {QR_code_latency}, \ndisplay latency {display_latency}")
    print(f"Display latency: {np.mean(display_latency)}")
    print(f"QR code latency: {np.mean(QR_code_latency)}")

if __name__ == "__main__":
    gc.disable()
    generate_qr_with_timestamp()
    gc.enable()