import numpy as np


class TaskHierarchyManager(object):
    def __init__(self, task, w_max, w_min, robot):
        self._task = task
        self._w_max = w_max
        self._w_min = w_min
        self._w_starting = self._task.w_hierarchy
        self._robot = robot
        self._start_time = 0.
        self._duration = 0.

    def initialize_ramp_to_min(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._w_starting = self._task.w_hierarchy

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._w_starting = self._task.w_hierarchy

    def update_ramp_to_min(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        """
        Problem #5
        ----------
        Set proper value for self._task.w_hierarchy. You would lienarly
        interpolate the value such that the weight is self._w_starting at
        self._start_time, and the weight is self._w_min at self._start_time +
        self._duration.

        Parameters to set
        -----------------
        self._task._w_hierarchy : Task weight needs to be set

        Parameters to use
        -----------------
        self._start_time : Interpolation start time
        self._w_starting : Initial time weight at self._start_time
        self._duration : Interpolation duration
        self._w_min : Final time weight at self._start_time + self._duration

        """
        slope = (self._w_starting-self._w_min)/self._duration
        self._task.w_hierarchy = -(slope*(t-self._start_time))+self._w_starting
        # print("made it to task update ramp min")


    def update_ramp_to_max(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        """
        Problem #6
        ----------
        Set proper value for self._task.w_hierarchy. You would lienarly
        interpolate the value such that the weight is self._w_starting at
        self._start_time, and the weight is self._w_max at self._start_time +
        self._duration.

        Parameters to set
        -----------------
        self._task._w_hierarchy : Task weight needs to be set

        Parameters to use
        -----------------
        self._start_time : Interpolation start time
        self._w_starting : Initial time weight at self._start_time
        self._duration : Interpolation duration
        self._w_max : Final time weight at self._start_time + self._duration

        """
        slope = (self._w_max-self._w_starting)/self._duration
        self._task.w_hierarchy = (slope*(t-self._start_time))+self._w_starting
        # print("made it to task update ramp max")

