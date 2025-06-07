from pickle import load

import numpy as np


class Stream:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.data = load(f)
        self.scans = self.data['scans']
        self.shifts = self.data['shifts']
        self.pose = np.zeros(3)
        self.odoms = self.data['odoms']
        self.i = 0
        self.max = len(self.scans)

    def __call__(self):
        if self.i < self.max:
            scan = self.scans[self.i]
            self.pose += self.shifts[self.i]
            odom = self.odoms[self.i]
            self.i += 1
            return scan, self.pose, odom
        else:
            return None, None, None