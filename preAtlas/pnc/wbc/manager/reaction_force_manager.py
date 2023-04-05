import numpy as np


class ReactionForceManager(object):
    def __init__(self, contact, maximum_rf_z_max, robot):
        self._contact = contact
        self._robot = robot
        self._maximum_rf_z_max = maximum_rf_z_max
        self._minimum_rf_z_max = 0.001
        self._starting_rf_z_max = contact.rf_z_max
        self._start_time = 0.
        self._duration = 0.

    def initialize_ramp_to_min(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._starting_rf_z_max = self._contact.rf_z_max

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._starting_rf_z_max = self._contact.rf_z_max

    def update_ramp_to_min(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        """
        Problem #3
        ----------
        Set proper value for self._contact.rf_z_max. You would lienarly
        interpolate the value such that the rf_z_max is self._starting_rf_z_max
        at self._start_time, and the weight is self._minimum_rf_z_max at
        self._start_time + self._duration.

        Parameters to set
        -----------------
        self._contact.rf_z_max : Upper bound for reaction force normal that
                                 needs to be set

        Parameters to use
        -----------------
        self._start_time : Interpolation start time
        self._starting_rf_z_max : Initial time upper bound at self._start_time
        self._duration : Interpolation duration
        self.self._minimum_rf_z_max : Final time upper bound at self._start_time
                                      + self._duration
        """
        # self.initialize_ramp_to_min(current_time,self._duration)
        slope = (self._starting_rf_z_max-self._minimum_rf_z_max)/self._duration
        self._contact.rf_z_max = -(slope*(t-self._start_time))+self._starting_rf_z_max
        # print("made it to force update ramp min")

    def update_ramp_to_max(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        """
        Problem #4
        ----------
        Set proper value for self._contact.rf_z_max. You would lienarly
        interpolate the value such that the rf_z_max is self._starting_rf_z_max
        at self._start_time, and the weight is self._maximum_rf_z_max at
        self._start_time + self._duration.

        Parameters to set
        -----------------
        self._contact.rf_z_max : Upper bound for reaction force normal that
                                 needs to be set

        Parameters to use
        -----------------
        self._start_time : Interpolation start time
        self._starting_rf_z_max : Initial time upper bound at self._start_time
        self._duration : Interpolation duration
        self.self._maximum_rf_z_max : Final time upper bound at self._start_time
                                      + self._duration

        """
        
        # self.initialize_ramp_to_max(current_time,self._duration)
        slope = (self._maximum_rf_z_max-self._starting_rf_z_max)/self._duration
        self._contact.rf_z_max = (slope*(t-self._start_time))+self._starting_rf_z_max
        
        
