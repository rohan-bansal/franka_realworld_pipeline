import threading
import time
from collections import deque

import numpy as np
from termcolor import cprint

from robomimic.utils.vis_utils import depth_to_rgb
import heapq

class ActionQueue:
    def __init__(self, max_len=60):
        self.heap = []
        self.counter = 0 # tie breaker for heap comparison, avoid comparing action
        self.max_len = max_len

    def add_action(self, time, action):
        """Add a (time, action) tuple to the queue."""
        heapq.heappush(self.heap, (time, self.counter, action))
        self.counter += 1
        if len(self.heap) > self.max_len:
            print(f"Warning: action queue exceed max len")

    def pop_next_action(self):
        """Pop and return the (time, action) tuple with the smallest time."""
        if self.heap:
            timestamp, _, action = heapq.heappop(self.heap)
            return timestamp, action
        else:
            raise IndexError("pop from an empty ActionQueue")

    def peek_next_action(self):
        """Peek at the (time, action) tuple with the smallest time without removing it."""
        if self.heap:
            return self.heap[0]
        else:
            raise IndexError("peek from an empty ActionQueue")

    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.heap) == 0

class Rate:
    def __init__(self, rate: float, name: str=None, log_warning=False):
        self.last = time.perf_counter()
        self.rate = rate
        self.dt   = 1.0 / self.rate
        self.name = name
        self.log_warning = log_warning

    def sleep(self) -> None:
        update_rate = 1.0 / (time.perf_counter() - self.last)
        # if self.name=="RL2 Robot Env":
        #     print(f"update rate is: {update_rate}")
        if update_rate < self.rate and self.log_warning:
            cprint(f"Warning: {self.name} update rate is {update_rate}Hz, lower than {self.rate}Hz", "red")
        while (self.last + 1.0 / self.rate) > time.perf_counter():
            time.sleep(0.001)
        self.last = time.perf_counter()

    # def sleep(self) -> None:
    #     time_to_sleep = (self.last + self.dt) - time.perf_counter()
    #     if time_to_sleep > 0:
    #         time.sleep(time_to_sleep)
    #     else:
    #         cprint(f"Warning: {self.name} update rate is {1.0/(time.perf_counter() - self.last)}Hz, lower than {self.rate}Hz", "red")
    #     self.last = time.perf_counter()

class ObsBufferUpdater:
    def __init__(self, update_function, name:str=None, rate:float=100.0, buffer_size:int=10):
        """
        To setup a sperate thread for update the observations (image, gripper) at a constant rate.
        Input: function to
        """
        # Store a reference to the camera object
        self.update_function = update_function
        # Initialize the deque buffer with a fixed size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self._rate = Rate(rate, name=name, log_warning=False)

        # Start the background thread to update the buffer with camera images
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._update_buffer)
        self._thread.daemon = True # NOTE: might not safe for termination and resources clean up
        self._thread.start()

        self.time_stamp = 0 # record the update time stamp for each image, for debugging. Maybe change to global timestamp later

    def _update_buffer(self):
        # A background function to update the buffer with the latest images
        while self._running:
            # Fetch an image from the camera and add it to the buffer
            obs =  self.update_function()

            if isinstance(obs, dict):
                obs['buffer_timestamp'] = np.array(self.time_stamp)
            self.time_stamp += 1
            with self._lock:
                self.buffer.append(obs)
            self._rate.sleep()

    def get_buffer(self, len: int=1):
        # Getter function to retrieve the buffer as a list of images
        # specify the len of the list (return the latest {len} observations
        if len > self.buffer_size:
            cprint(f"Warning: trying to getting buffer list larger than size {self.buffer_size}, returning full buffer", "red")
            len = self.buffer_size
        elif len < 1:
            len = 1

        with self._lock:
            return list(self.buffer)[-len:]

    def stop(self):
        # Stop the background thread
        self._running = False
        time.sleep(0.2)
        print(f"deleting thread {self._thread}")
        self._thread.join()


    def __del__(self):
        self.stop()

if __name__=="__main__":
    # Example usage
    queue = ActionQueue()
    queue.add_action(10.999, "Action A")
    queue.add_action(50.444, "Action B")
    queue.add_action(8, "Action C")

    print(queue.pop_next_action())  # Output: (5, 'Action B')
    print(queue.pop_next_action())  # Output: (8, 'Action C')
    print(queue.pop_next_action())  # Output: (10, 'Action A')
