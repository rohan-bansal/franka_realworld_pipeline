from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import collections

class ObsAccumulator:
    def __init__(self):
        self.data = collections.defaultdict(list)
        self.timestamps = collections.defaultdict(list)

    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """
        data:
            key: T,*
        """
        for key, value in data.items():
            for i, t in enumerate(timestamps):
                if (key not in self.timestamps) or (self.timestamps[key][-1] < t):
                    self.timestamps[key].append(t)
                    self.data[key].append(value[i])