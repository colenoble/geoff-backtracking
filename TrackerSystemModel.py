import pandas as pd
import numpy as np
import datetime

import math

import core
import configuration
from configuration import SAT2, SATSloped
from utilities import Location2

import traceback

class TrackerSystemModel():

    def __init__(self, lat, lon, tracker_width, tracker_pitch, tracker_phi, timezone):
        self.lat = lat
        self.lon = lon
        self.tracker_width = tracker_width
        self.tracker_pitch = tracker_pitch
        self.tracker_phi = tracker_phi
        self.timezone = timezone
        self.location = Location2(lat, lon, 0.0, timezone)
        self.tracker = SAT2(self.location, tracker_width, tracker_phi, tracker_pitch, 0)
        #self.stm = SeparationTranspositionModel(self.location)
        self.b0 = 0.05

    def getTracker(self):
        return self.tracker

    def getTrackerConfiguration(self, time):
        return self.tracker.getTrackerConfiguration(time)

    # This gives the angle of the sun in the panel plane if alpha is the angle of the panel in radian, with alpha positive if tilted clockwise looking from south to north
    def getSunsNormalInPlane(self, time, delta):
        (day, back_tracking, elevation, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        alpha = panel_angle + delta
        a1 = math.asin(math.cos(theta) * math.cos(phi))
        a2 = math.atan(math.tan(theta) / math.sin(phi))
        eta = math.atan((math.cos(a1) * math.sin(a2 + alpha)) / (
                    (math.cos(a1)) ** 2.0 * (math.cos(a2 + alpha)) ** 2.0 + (math.cos(theta)) ** 2.0 * (
                math.cos(phi)) ** 2.0) ** 0.5)
        return math.fabs(math.sin(eta))

    def getSunsPerpendicularComponent(self, time, panel_angle ):
        (day, back_tracking, elevation, optimum_panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        #print( '{}, {}, {}'.format( panel_angle, phi, theta ) )
        return math.sin(panel_angle) * math.cos(theta) * math.sin(phi) + math.cos(panel_angle) * math.sin(theta)

    def getSunsOptimumNormalInPlane(self, time):
        (day, back_tracking, elevation, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        alpha = panel_angle
        a1 = math.asin(math.cos(theta) * math.cos(phi))
        a2 = math.atan(math.tan(theta) / math.sin(phi))
        eta = math.atan((math.cos(a1) * math.sin(a2 + alpha)) / (
                    (math.cos(a1)) ** 2.0 * (math.cos(a2 + alpha)) ** 2.0 + (math.cos(theta)) ** 2.0 * (
                math.cos(phi)) ** 2.0) ** 0.5)
        return math.fabs(math.sin(eta))

    def backtracking(self, time):
        (day, back_tracking, elevation, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        if day:
            return back_tracking
        else:
            return False

    def panel_angle(self, time):
        (day, back_tracking, elevation, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        return panel_angle

    def sun_angle_in_panel_plane(self, time):
        (day, back_tracking, elevation, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        return elevation

    #Commenting this out for now, not needed for angle calcualtions
    #def getBeamDiffuse(self, ghi, altitude):
    #    return self.stm.separationModel(ghi, altitude)

    def getIAMLoss(self, elevation):
        iam_loss = 1.0
        if elevation > math.pi / 2.0 - math.acos(1 / (1 / self.b0 + 1)):
            iam_loss = self.b0 * (1 / math.cos(math.pi / 2.0 - elevation) - 1.0)
        return iam_loss

class TrackerSystemModelSloped():

    def __init__(self, lat, lon, tracker_width, tracker_pitch, tracker_phi, ground_slope, timezone):
        self.lat = lat
        self.lon = lon
        self.tracker_width = tracker_width
        self.tracker_pitch = tracker_pitch
        self.tracker_phi = tracker_phi
        self.ground_slope = ground_slope
        self.timezone = timezone
        self.location = Location2(lat, lon, 0.0, timezone)
        self.tracker = SATSloped(self.location, tracker_width, tracker_phi, tracker_pitch, ground_slope)
        #self.stm = SeparationTranspositionModel(self.location)
        self.b0 = 0.05

    def getTracker(self):
        return self.tracker

    def getTrackerConfiguration(self, time):
        return self.tracker.getTrackerConfiguration(time)

    # This gives the angle of the sun in the panel plane if alpha is the angle of the panel in radian, with alpha positive if tilted clockwise looking from south to north
    def getSunsNormalInPlane(self, time, delta):
        (day, back_tracking, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        alpha = panel_angle + delta
        a1 = math.asin(math.cos(theta) * math.cos(phi))
        a2 = math.atan(math.tan(theta) / math.sin(phi))
        eta = math.atan((math.cos(a1) * math.sin(a2 + alpha)) / (
                    (math.cos(a1)) ** 2.0 * (math.cos(a2 + alpha)) ** 2.0 + (math.cos(theta)) ** 2.0 * (
                math.cos(phi)) ** 2.0) ** 0.5)
        return math.fabs(math.sin(eta))

    def getSunsPerpendicularComponent(self, time, panel_angle ):
        (day, back_tracking, optimum_panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        #print( '{}, {}, {}'.format( panel_angle, phi, theta ) )
        return math.sin(panel_angle) * math.cos(theta) * math.sin(phi) + math.cos(panel_angle) * math.sin(theta)

    def getSunsOptimumNormalInPlane(self, time):
        (day, back_tracking, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        alpha = panel_angle
        a1 = math.asin(math.cos(theta) * math.cos(phi))
        a2 = math.atan(math.tan(theta) / math.sin(phi))
        eta = math.atan((math.cos(a1) * math.sin(a2 + alpha)) / (
                    (math.cos(a1)) ** 2.0 * (math.cos(a2 + alpha)) ** 2.0 + (math.cos(theta)) ** 2.0 * (
                math.cos(phi)) ** 2.0) ** 0.5)
        return math.fabs(math.sin(eta))

    def backtracking(self, time):
        (day, back_tracking, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        if day:
            return back_tracking
        else:
            return False

    def panel_angle(self, time):
        (day, back_tracking, panel_angle, theta, phi) = self.getTrackerConfiguration(time)
        return panel_angle

    #Commenting this out for now, not needed for angle calcualtions
    #def getBeamDiffuse(self, ghi, altitude):
    #    return self.stm.separationModel(ghi, altitude)

    def getIAMLoss(self, elevation):
        iam_loss = 1.0
        if elevation > math.pi / 2.0 - math.acos(1 / (1 / self.b0 + 1)):
            iam_loss = self.b0 * (1 / math.cos(math.pi / 2.0 - elevation) - 1.0)
        return iam_loss