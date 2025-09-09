from builtin_interfaces.msg._time import Time

def ros_time_to_float(ros_time):
    seconds = ros_time.sec
    nanoseconds = ros_time.nanosec
    float_time = seconds + nanoseconds/(1e+9)
    return float_time


def float_time_to_ros(float_time):
    sec = int(float_time)
    nanosec = int((float_time - sec) * 1e9)
    ros_time = Time(sec=sec, nanosec=nanosec)
    return ros_time