from datetime import datetime
import numpy as np
import math
from scipy.signal import *
from scipy.ndimage import filters
from scipy import interpolate
import time
import pytz

import itertools
'''
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
'''
import pandas as pd
import pyproj

import requests

import os.path

import seaborn as sns
sns.set_color_codes()

from core import Model

# from suncalc import get_position

# Using the subprocess module allows us to call Radiance external commands and return the stdout from the subprocess to Python
from subprocess import PIPE, run

# Method to retrieve the stdout from the subprocess
def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

class NewtonsMethod():
    
    def __init__(self, f, dfdx, x0, tol):
        self.f = f
        self.dfdx = dfdx
        self.x0 = x0
        self.tol = tol
        
    def solve(self):
        fx0 = 2*self.tol
        xn = self.x0
        count = 0
        #while abs(fx0) > self.tol:
        while count < 100:
            count += 1
            xn = xn - self.f(xn) / self.dfdx(xn)
            fx0 = self.f(xn)
        return xn        

class SmoothingFilter():
    # Convenience method to smooth a time series using a convolution with a Gaussian kernel
    # User sets the windowFraction, the fraction of the total time series that the convolving window will be in length
    # Also set is the kernelFraction, the fraction of the convolving window width that the standard deviation of the Gaussian will be

    def __init__(self, windowFraction, kernelFraction):
        self.windowFraction = windowFraction
        self.kernelFraction = kernelFraction

    def smoothSeries(self, x):
        x = np.asarray(list(x))
        seriesLength = len(x)
        window = int(seriesLength / self.windowFraction) + 1
        kernel = int(window / self.kernelFraction) + 1
        gaussKernel = gaussian(window, kernel)
        normedKernel = gaussKernel / gaussKernel.sum()
        outputSeries = filters.convolve1d(x, normedKernel)
        return outputSeries

class Location():
    # This class represents the location where the solar geometry is to be determined
    # Recorded is the lat and lon
    # Before use, it is necessary to set the time zone offset from GMT, with positive offsets east of Greenwich, so -5 hours for EST for example
    # Also necessary to set for this class is the startMonth, startDay, endMonth and endDay of Daylight Savings Time for the location
    # The method isDST returns then, for a particular time, whether or not a DST correction should be applied

    def __init__(self, lat, lon, elev):
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.start = None
        self.end = None
        if lat >= 0.0:
            epsg = int((self.lon + 180.0)/6.0) + 32601
        else:
            epsg = int((self.lon + 180.0) / 6.0) + 32701
        self.source_proj = pyproj.Proj("+init=EPSG:4326")
        self.target_proj = pyproj.Proj("+init=EPSG:{}".format(epsg))
        self.easting, self.northing = self.projectToUTM(self.lat, self.lon)
        
    # For some countries daylight savings rules do not follow a typical pattern, e.g. Morocco where, to accomodate Ramadan,
    # time jumps behind instead of ahead for the first jump of the year, in May and the jumps back in June
    # Also, daylight savings is not observed every year in many countries as well.
    # For this reason, we need a provision to manually set the offset so that the isDST method can be over-ridden when DST
    # is changing arbitarily.
    
    # This is where the time-zone offset from GMT is defined.
    # Note that everything east of GMT has a positive offset, so Eastern Time has an offset of -5 hours
    def setTimeZoneOffsetFromGMT(self, offset):
        self.offset = offset

    # Get the offset from GMT accounting for Daylight Savings Time
    def getGMTOffset(self, time):
        offset = self.offset
        if self.isDST(time):
            offset = offset + 1
        return offset

    # Set the start and end times of Daylight Savings Time
    def setDST(self, startMonth, startDay, endMonth, endDay):
        self.startMonth = startMonth
        self.startDay = startDay
        self.endMonth = endMonth
        self.endDay = endDay

    # A method to determine if it is Daylight Savings Time
    def isDST(self, time):
        # For a particular time, is is daylight savings time?
        year = time.year
        if not self.start:
            self.start = datetime(year, self.startMonth, self.startDay, 0, 0, 0).timetuple().tm_yday
            self.end = datetime(year, self.endMonth, self.endDay, 0, 0, 0).timetuple().tm_yday
        isDST = False
        day = time.timetuple().tm_yday
        if day > self.start and day <= self.end:
            isDST = True
        return isDST

    # This method returns the UTM coordinates that correspond to the Location object for the 
    # input latitude and longitude. Note that if the latitude and the longitude are not in the same
    # UTM zone, then distortion may result. No check is made for this, it is assumed the inputs are close by.
    def projectToUTM(self, longitude, latitude):
        return pyproj.transform(self.source_proj, self.target_proj, latitude, longitude)
        
    # Fix this, there is no decent API so create some code to download NetCDF data from a weather service
    # and do the interpolation myself. All the APIs want the user to input a city name, not coordiates so
    # they are useless for this sort of application. Pressure is not so important anyway for the irradiation calculations
    def getPressure(self):
        return 101000.0

class Location2():
    
    # An improvement on Location, as this class uses the pytz library.
    # Timezones are very complicated, for instance Morocco has the time jump backward first in May, and forward in June
    # to accomodate Ramadan, while other locations jump forward in Spring and back in Fall for agriculture.
    # In many places, time-zones are not observed every year as well, making this even more complicated.
    # To get a list of timezone codes
    #
    # MAKE SURE THAT THE PYTZ LIBRARY IS THE LATEST VERSION
    # 
    # Use: conda install -c cefca pytz
    # An older version I used did not have correct daylight savings time settings for Morocco, which was only fixed in a newer version
    
    # 'timezone' is a string of one of the official timezone codes, like 'America/Toronto'. To get a list of these, just use:
    #
    # for tz in pytz.all_timezones:
    #     print(tz)
    #
    def __init__(self, lat, lon, elev, timezone):
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.start = None
        self.end = None
        if lat >= 0.0:
            epsg = int((self.lon + 180.0)/6.0) + 32601
        else:
            epsg = int((self.lon + 180.0) / 6.0) + 32701
        self.source_proj = pyproj.Proj("+init=EPSG:4326")
        self.target_proj = pyproj.Proj("+init=EPSG:{}".format(epsg))
        self.easting, self.northing = self.projectToUTM(self.lat, self.lon)
        self.timezone = timezone
        self.local = pytz.timezone(timezone)
        self.utc = pytz.timezone('UTC')
        self.fmt = '%Y-%m-%d %H:%M:%S %Z%z'
        
    def getUTCTime(self, time):
        local_time = self.local.localize(time)
        utc_time = local_time.astimezone(self.utc)
        return utc_time
     
    # This method returns the UTM coordinates that correspond to the Location object for the 
    # input latitude and longitude. Note that if the latitude and the longitude are not in the same
    # UTM zone, then distortion may result. No check is made for this, it is assumed the inputs are close by.
    def projectToUTM(self, longitude, latitude):
        return pyproj.transform(self.source_proj, self.target_proj, latitude, longitude)
        
    # Fix this, there is no decent API so create some code to download NetCDF data from a weather service
    # and do the interpolation myself. All the APIs want the user to input a city name, not coordiates so
    # they are useless for this sort of application. Pressure is not so important anyway for the irradiation calculations
    def getPressure(self):
        return 101000.0
    
class SolarPositionCalculator():
    # This class returns the altitude and azimuth of the sun for a particular location given a datetime object
    # corresponding to the time of measurement

    def __init__(self, location):
        self.location = location
        self.dtr = math.pi / 180.0

    def getGeometry(self, time):
        self.time = time
        day = time.timetuple().tm_yday

        gmt_offset = self.location.getGMTOffset(time)

        # Get the time in hours past midnight in GMT time
        hour = time.hour + time.minute / 60.0 + time.second / 3600.0 - gmt_offset
        # Find the number of leap years since 1949
        delta = time.year - 1949
        leap = int(delta / 4.0)
        # From this, find the Julian day value corresponding to the time
        julian = 32916.5 + delta * 365.0 + leap + day + hour / 24.0

        # With the Julian time, find the Julian time since January 1, 2000
        # This is the start date from which the following equations are defined in the Astronomer's Almanac
        # So find first the Julian time since Jan. 1, 2000
        time = julian - 51545.0

        # These are intermediate data points needed to calculate the celestial coordinates of the sun's position
        # mnl -> mean longitude
        mnl = 280.460 + 0.9856474 * time
        mnl = mnl % 360

        # mna -> mean anomoly
        mna = 357.528 + 0.9856003 * time
        mna = mna % 360
        mna = mna * self.dtr

        # ecl -> ecliptic longitude, obl -> obliquity of ecliptic
        ecl = mnl + 1.915 * math.sin(mna) + 0.020 * math.sin(2 * mna)
        ecl = ecl % 360
        obl = 23.439 - 0.0000004 * time
        ecl = ecl * self.dtr
        obl = obl * self.dtr

        # Calculate celestial coordinates
        # ra -> right ascension, dec -> declination
        # This is the right ascension in radians
        ra = math.atan2(math.cos(obl) * math.sin(ecl), math.cos(ecl))
        dec = math.asin(math.sin(obl) * math.sin(ecl))
        self.declination = dec

        # Local coordinates
        # gmst -> Greenwich mean sideral time
        gmst = 6.697375 + 0.0657098242 * time + hour
        gmst = gmst % 24

        # lmst -> local mean sideral time
        lmst = gmst + self.location.lon / 15.0
        lmst = lmst % 24
        lmst = lmst * 15.0 * self.dtr

        # ha -> hour angle
        ha = lmst - ra
        if ha > 2.0 * math.pi:
            ha = ha - 2.0 * math.pi

        # With the above calculated, find the elevation and azimuth from earth
        # el -> solar elevation
        # ha -> azimuth - defined as the angle clockwise from north in accordance with the NOAA equations
        el = math.asin(math.sin(dec) * math.sin(self.location.lat * self.dtr) + math.cos(dec) * math.cos(
            self.location.lat * self.dtr) * math.cos(ha))
        az = math.pi - math.asin(-1.0 * math.cos(dec) * math.sin(ha) / math.cos(el))
        return (el, az)

    # Convenience method to return the elevation and azimuth as angles, not radians
    def getGeometryAsAngles(self, time):
        el, az = self.getGeometry(time)
        return (el / self.dtr, az / self.dtr)

class SolarPositionCalculator2():
    # This class returns the altitude and azimuth of the sun for a particular location given a datetime object
    # corresponding to the time of measurement
    
    # It is an improvement on the SolarPositionCalculator since it can use the improved Location2 class which has time-zone
    # aware intelligence gained from the pytz library
    
    def __init__(self, location):
        self.location = location
        self.dtr = math.pi / 180.0

    def getGeometry(self, time):
        self.time = time
        local_time = time
        day = time.timetuple().tm_yday
        
        # Use the location's time-zone aware method to grab the UTC time corresponding to the location time sent
        time = self.location.getUTCTime(time)

        # Get the time in hours past midnight in GMT time
        hour = time.hour + time.minute / 60.0 + time.second / 3600.0
        # Find the number of leap years since 1949
        delta = time.year - 1949
        leap = int(delta / 4.0)
        # From this, find the Julian day value corresponding to the time
        julian = 32916.5 + delta * 365.0 + leap + day + hour / 24.0

        # With the Julian time, find the Julian time since January 1, 2000
        # This is the start date from which the following equations are defined in the Astronomer's Almanac
        # So find first the Julian time since Jan. 1, 2000
        time = julian - 51545.0

        # These are intermediate data points needed to calculate the celestial coordinates of the sun's position
        # mnl -> mean longitude
        mnl = 280.460 + 0.9856474 * time
        mnl = mnl % 360

        # mna -> mean anomoly
        mna = 357.528 + 0.9856003 * time
        mna = mna % 360
        mna = mna * self.dtr

        # ecl -> ecliptic longitude, obl -> obliquity of ecliptic
        ecl = mnl + 1.915 * math.sin(mna) + 0.020 * math.sin(2 * mna)
        ecl = ecl % 360
        obl = 23.439 - 0.0000004 * time
        ecl = ecl * self.dtr
        obl = obl * self.dtr

        # Calculate celestial coordinates
        # ra -> right ascension, dec -> declination
        # This is the right ascension in radians
        ra = math.atan2(math.cos(obl) * math.sin(ecl), math.cos(ecl))
        dec = math.asin(math.sin(obl) * math.sin(ecl))
        self.declination = dec

        # Local coordinates
        # gmst -> Greenwich mean sideral time
        gmst = 6.697375 + 0.0657098242 * time + hour
        gmst = gmst % 24

        # lmst -> local mean sideral time
        lmst = gmst + self.location.lon / 15.0
        lmst = lmst % 24
        lmst = lmst * 15.0 * self.dtr

        # ha -> hour angle
        ha = lmst - ra
        if ha > 2.0 * math.pi:
            ha = ha - 2.0 * math.pi

        # With the above calculated, find the elevation and azimuth from earth
        # el -> solar elevation
        # ha -> azimuth - defined as the angle clockwise from north in accordance with the NOAA equations
        el = math.asin(math.sin(dec) * math.sin(self.location.lat * self.dtr) + math.cos(dec) * math.cos(
            self.location.lat * self.dtr) * math.cos(ha))
        az = math.pi - math.asin(-1.0 * math.cos(dec) * math.sin(ha) / math.cos(el))
        return (el, az)

    # Convenience method to return the elevation and azimuth as angles, not radians
    def getGeometryAsAngles(self, time):
        el, az = self.getGeometry(time)
        return (el / self.dtr, az / self.dtr)

class SolarGeometry():

    # This class corrects for alignment errors of panels in the EW direction - where the rotation axis is along a NS axis -
    # by returning the altitude and azimuth of the sun relative to the rotated panel - in other words, the altitude and azimuth
    # in the rotated panel's coordinate system. It also determines the relative altitude and azimuth for a fixed tilt panel

    def __init__(self):
        self.dtr = math.pi / 180.0

    def reorientForEWRotation(self, alt, azi, alpha):

        # This method takes and altitude and an azimuth and a horizontal panel that has a slight rotation of angle alpha in the east-west direction (i.e. a rotation along a NS axis)
        # It returns what the altitude and azimuth would be relative to the panel that has this rotation of alpha in the east-west direction
        # In other words, what the altitude and azimuth would be in the rotated panels coordinate system

        altp = 0.0
        azip = 0.0
        if alt > 0:
            alpha = alpha * self.dtr
            y = np.abs(1.0 / (math.tan(alt) * (1.0 + (math.tan(azi)) ** 2.0) ** 0.5)) * np.sign(math.cos(azi))
            x = np.abs(math.tan(azi) * y) * np.sign(math.sin(azi))
            z = 1.0
            beta = math.atan2(z, x)
            l = (1.0 + x ** 2.0) ** 0.5
            betap = beta + alpha
            xp = l * math.cos(betap)
            zp = l * math.sin(betap)
            yp = y
            altp = math.atan2(zp, (xp ** 2.0 + yp ** 2.0) ** 0.5)
            azip = math.atan2(xp, yp)
            if azip < 0:
                azip = azip + 2.0 * math.pi
        return altp, azip

    def getGeometryInTiltPlane(self, alt, azi, tilt):

        # This method takes an altitude and azimuth defined for a horizontal panel and determines the altitude and azimuth in the tilted panels coordinate system
        # alt and azi are defined in radians, but the tilt angle is in degrees

        altr = 0.0
        azir = 0.0
        if alt > 0:
            tilt = tilt * self.dtr
            y = np.abs(1.0 / (math.tan(alt) * (1.0 + (math.tan(azi)) ** 2.0) ** 0.5)) * np.sign(math.cos(azi))
            x = np.abs(math.tan(azi) * y) * np.sign(math.sin(azi))
            z = 1.0
            alpha = math.atan2(z, y)
            alphap = alpha - tilt
            rho = (z ** 2.0 + y ** 2.0) ** 0.5
            yp = rho * math.cos(alphap)
            zp = rho * math.sin(alphap)
            xp = x
            xr = xp / zp
            yr = yp / zp
            zr = 1.0
            altr = math.atan2(zr, (xr ** 2.0 + yr ** 2.0) ** 0.5)
            azir = math.atan2(xr, yr)
            if azir < 0:
                azir = azir + 2.0 * math.pi
        return altr, azir

class SeparationTranspositionModel():
    def __init__(self, location):
        self.sg = SolarGeometry()
        self.loc = location
        self.spc = SolarPositionCalculator(self.loc)
        self.SolarConstant = 1367.0

    def getExtraterrestrialRadiation(self, date):
        # This method returns the extraterrestrial radiation for a particular time
        # The extraterrestrial radiation is the radiation that would be received at the top of the atmosphere
        # if there were no atmospheric effects. It is calculated using the following equation:
        # Io = Io0 * (1.00011 +  0.034211 * cos(2 * pi * day / 365) 0.00128 * sin(2 * pi * day / 365) )
        # where Io0 is the solar constant, day is the day of the year, and Io is the extraterrestrial radiation
        # The value of Io0 is set to 1367 W/m^2

        day = date.timetuple().tm_yday
        Io = self.SolarConstant * ( 1.00011 + 0.034211 * math.cos(2.0 * math.pi * day / 365.0) + 0.00128 * math.sin(2.0 * math.pi * day / 365.0) )
        return Io

    def ErbsSeparationModel(self, Gh, altitude,time):
        # Based on the EKD/Erbs (1982) Model
        # This is a method to separate out the diffuse and direct components of irradiation based upon the Erbs model
        # The inputs are the global horizontal irradiation and the altitude angle of the sun from a horizontal surface in radians
        # Gh -> global horizontal irradiation, Id -> diffuse irradiation, Ib -> beam irradiation
        # Io -> solar constant irradiation, Kt -> clearness index, Kd -> diffuse fraction
        # Kt = Gh/Io, Kd = Id/Gh
        # Make sure that cloud edge spikes don't screw up the result. Limit Gh to Io at maximum, so Kt to 1.0 at maximum.
        E0 = self.getExtraterrestrialRadiation(time)
        altitude = max(altitude * math.pi / 180, 0.0)
        Kt = Gh / ( E0 * math.sin(altitude) )  
        Kd = 0.0
        if Kt <= 0.22:
            Kd = 1.0 - 0.09 * Kt
        elif Kt > 0.22 and Kt <= 0.8:
            Kd = 0.9511 - 0.1640 * Kt + 4.388 * Kt ** 2.0 - 16.638 * Kt ** 3.0 + 12.336 * Kt ** 4.0
        elif Kt > 0.8:
            Kd = 0.165
        Ib = 0
        Id = 0
        if altitude > 0:
        # Now with the diffuse fraction calculated, we get the beam component     
        # Now with the diffuse fraction calculated, we get the beam component
            Ib = Gh * (1.0 - Kd) / math.sin(altitude)
            Id = Gh * Kd
        return (Ib, Id)
    
    def ReindlSeparationModel(self, Gh, altitude, time):
        # Based on the Reindl (1990) Model
        # This is a method to separate out the diffuse and direct components of irradiation based upon the Erbs model
        # The inputs are the global horizontal irradiation and the altitude angle of the sun from a horizontal surface in radians
        # Gh -> global horizontal irradiation, Id -> diffuse irradiation, Ib -> beam irradiation
        # Io -> solar constant irradiation, Kt -> clearness index, Kd -> diffuse fraction
        # Kt = Gh/Io, Kd = Id/Gh
        # Make sure that cloud edge spikes don't screw up the result. Limit Gh to Io at maximum, so Kt to 1.0 at maximum.
        E0 = self.getExtraterrestrialRadiation(time)
        altitude = max(altitude * math.pi / 180, 0.0)
        Kt = Gh / ( E0 * math.sin(altitude) )  
        Kd = 0.0
        if Kt <= 0.3:
            Kd = 1.02 - 0.248 * Kt  
        elif Kt > 0.3 and Kt <= 0.78:
            Kd = 1.45 - 1.67 * Kt 
        elif Kt > 0.78:
            Kd = 0.147
        Ib = 0
        Id = 0
        if altitude > 0:
        # Now with the diffuse fraction calculated, we get the beam component        
            Ib = Gh * (1.0 - Kd) / math.sin(altitude)
            Id = Gh * Kd
        return (Ib, Id)

    def transpositionModel(self, Ib, Id, altitude, tilt):
        # Returns the global irradiation for a surface at an angle of altitude (radians) from the sun
        # and with the surface of the panel at an angle of tilt (degrees)
        # Ib -> beam irradiation, Id -> diffuse irradiation
        # altitude -> altitude of sun from surface, tilt -> tilt of surface from horizontal
        # Note that the diffuse part cannot be taken as the full diffuse irradiation of the sky
        # since part of the sky is behind the tilt plane of the panel.
        return Ib * math.sin(altitude) + Id * (1.0 + math.cos(tilt * math.pi / 180.0)) / 2.0

    def separateSeries(self, Ghs, times):
        # This method takes a list of Gh values and corresponding times and returns the separated irradiances for the location
        separatedSeries = []
        for data in zip(Ghs, times):
            Gh = data[0]
            time = data[1]
            altitude, azimuth = self.spc.getGeometry(time)
            Ib, Id = self.ReindlSeparationModel(Gh, altitude, time)
            separatedSeries.append([Ib, Id])
        return separatedSeries

    def transposeSeries(self, Ibs, Ids, times, tilt):
        transposedSeries = []
        for data in zip(Ibs, Ids, times):
            Ib = data[0]
            Id = data[1]
            time = data[2]
            alt, azi = self.spc.getGeometry(time)
            talt, tazi = self.sg.getGeometryInTiltPlane(alt, azi, tilt)
            Gh = self.transpositionModel(Ib, Id, talt, tilt)
            transposedSeries.append(Gh)
        return transposedSeries

    def separateSeriesWithEWError(self, Ghs, times, error):
        separatedSeries = []
        for data in zip(Ghs, times):
            beam = 0.0
            diffuse = 0.0
            Gh = data[0]
            time = data[1]
            altitude, azimuth = self.spc.getGeometry(time)
            alt, azi = self.sg.reorientForEWRotation(altitude, azimuth, error)
            Ib, Id = self.separationModel(Gh, alt)
            separatedSeries.append([Ib, Id])
        return separatedSeries

    def transposeSeriesWithError(self, Ibs, Ids, times, tilt, error):
        transposedSeries = []
        for data in zip(Ibs, Ids, times):
            Ib = data[0]
            Id = data[1]
            time = data[2]
            altitude, azimuth = self.spc.getGeometry(time)
            alt, azi = self.sg.getGeometryInTiltPlane(altitude, azimuth, tilt)
            talt, tazi = self.sg.reorientForEWRotation(alt, azi, error)
            tglobal = self.transpositionModel(Ib, Id, talt, tilt)
            transposedSeries.append(tglobal)
        return transposedSeries

class PyranometerTiltCorrector():
    # This class will take a series of irradiances (irrads) from a pyranometer with a given tilt to the horizontal, and correct for any
    # errors in alignment of the pyranometer in the EW direction, i.e. an error that leaves the solar noon peak of irradiance
    # offset from the true solar noon. Since an Erb's based separation model is used, it is necessary to also give horizontal
    # irradiances (hirrads) as well and the times corresponding to both irradiances (times). The tilt angle (tilt) is also given as well
    # as the angular offset (correction) necessary to re-orient the pyranometer.

    def __init__(self, location):
        self.location = location
        self.stm = SeparationTranspositionModel(self.location)
        self.sf = SmoothingFilter(6, 6)

    def correctSeries(self, hirrads, irrads, times, tilt, correction):
        # First, smooth the provided horizontal series and the tilted series.
        # Use the smoothed series to calculate the expected smoothed tilted irradiances
        # from the smoothed horizontal series with the known tilt, using the
        # SeparationTransposition model. We use smoothed series so that irradiance spike
        # noise add any additional errors from the noise - we just want the mean behaviour.
        # Since the separation and transposition models are not perfect, the calculated tilted
        # irradiance will not match the actual tilted irradiance exactly, due to model errors,
        # so the ratio between the two is taken to allow for a one-time library of ratios,
        # specific to the day the time series were taken from, to fix any later-calculated tilted irradiances.
        # Next the direct and diffuse irradiations are separated out from horizontal irradiances,
        # and the tilt of the pyranometer is re-oriented by the given correction value.
        # The two values are then transposed back together giving an expected tilted irradiance now correctly
        # oriented. As the separation/transposition models are not perfect, the resulting tilted irradiance will
        # not have the correct magnitudes, so the fixing ratios determined above are then used to correct the
        # tilted irradiance magnitudes. The end result will be a irradiance time series with the irradiances re-oriented
        # applying the tilt correction, with values scaled to have the correct magnitudes.
        #
        # So,first smooth both series using a Gaussian filter
        # The smoothing is necessary to reduce the noise of irradiance spikes
        smoothed_hirrads = self.sf.smoothSeries(hirrads)
        smoothed_irrads = self.sf.smoothSeries(irrads)
        # Now, get the beam (Ib) and diffuse (Id) irradiances from the smoothed horizontal irradiances using the
        # separation model and then save these as individual series to pass to the transposition model
        output = self.stm.separateSeries(smoothed_hirrads, times)
        Ibs = [a[0] for a in output]
        Ids = [a[1] for a in output]
        # Now transpose these back together to give the expected tilt irradiances
        calculated_irrads = self.stm.transposeSeries(Ibs, Ids, times, tilt)
        # What we want from this is how the corrected tilt irradiances compare to the actual values - a ratio between the two
        # that will allow us to fix any errors resulting from the use of the separation and transposition models.
        ratios = smoothed_irrads / calculated_irrads
        # Now, take these ratios and use them to create an interpolation function fixing_ratios that we will use later on to
        # retrieve the fixing ratios for any calculated tilt irradiances.
        # However, before we do this, we will have to remember to sort our ratios (our function y values) and our calculated_irrads
        # (our x values) so that the x values are strictly increasing - this is required by the interpolation routine.
        # Also, make sure that (x,y) = (0,0) is inserted at the start of the list, and (x,y) = (1367.0,1.0) is inserted at the end
        # so that the interpolation routine will not throw any 'out of interpolation bounds' errors when we try to use it later.
        # So first sort the two by the calculated_irrads, zipping together so that they stay associated
        sortList = sorted(zip(calculated_irrads, ratios), key=lambda pair: pair[0])
        # Then pull them apart to use in the interpolation function
        sorted_irrads = np.asarray([item[0] for item in sortList])
        sorted_ratios = np.asarray([item[1] for item in sortList])
        # Add in our start and end values so that we have our bounds set to avoid any out of bounds errors later
        sorted_irrads = np.append([0.0], sorted_irrads)
        sorted_ratios = np.append([0.0], sorted_ratios)
        sorted_irrads = np.append(sorted_irrads, [1367.0])
        sorted_ratios = np.append(sorted_ratios, [1.0])
        # Now, create the function
        fixing_ratios = interpolate.interp1d(sorted_irrads, sorted_ratios)
        # With our fixing ratio function available, we can now go ahead to re-orient the time series.
        # First, separate out the beam and diffuse components as above, but this time with the real series.
        output = self.stm.separateSeries(hirrads, times)
        Ibs = [a[0] for a in output]
        Ids = [a[1] for a in output]
        # Now, transpose these back together, but this time with the angle correction applied to the tilted panel
        reoriented_irrads = self.stm.transposeSeriesWithError(Ibs, Ids, times, tilt, correction)
        # As we know, this series will have the right orientation, but the error introduced by the STM model, so
        # use our calculated fixing ratios to fix this.
        f_ratios = fixing_ratios(reoriented_irrads)
        # Finally, use these ratios to fix our reoriented_irrads
        reoriented_irrads = reoriented_irrads * f_ratios
        return reoriented_irrads

class IAMModel():

    '''
    This class determines the Incidence Angle Modifier (IAM) for a panel at a given tilt at any time by
    calculating the solar position and then determining what the angle that the sun makes with the tilted panel.

    The equation that this model uses to calculate the IAM is the ASHRAE model:

    F = 1 - b0 * ( 1/cos(theta) - 1)      where theta is the angle of the sun with the panel plane, and b0 is a parameter

    Use b0 = 0.05 for normal glass, and b0 = 0.04 for anti-reflective (AR) glass

    Note that tilt is defined in degrees, not radians, and location is a Location object.

    TODO: This model makes an assumption which is too simple and which should be worked on.
    It assumes that all of the sun's light is direct, with the angle determined by the sun's position,
    so it ignores the effect of IAM modifiers on more isotropically distributed diffuse light.

    This model should also be based upon Fresnel's equations. The ASHRAE equations are not accurate enough
    and doing the Fresnels calculations are easily done, so these should be used in a later version.

    What this model will also do in a later version is calculate the difference between the clear sky component of light
    and the actual to determine the amount of scattering. The IAM will then be used on the direct component and an
    integral over all directions will be used on the diffuse component.

    For this, it will be necessary to create a SeparationModel to infer the direct and diffuse components of light
    from the measured POA and the sun's position. Then, a TranspositionModel will be needed to add the components
    back together after applying an IAM correction to each component.

    '''

    def __init__(self, b0, location, tilt):
        self.b0 = b0
        self.tilt = tilt
        self.spc = SolarPositionCalculator(location)
        self.sg = SolarGeometry()

    def getIAM(self, time):
        dtr = self.spc.dtr
        el, az = self.spc.getGeometry(time)
        elt, azt = self.sg.getGeometryInTiltPlane(el,az,self.tilt)
        if elt < math.acos(1/(1/self.bo + 1)):
            return 0.0
        else:
            return 1.0 - self.b0 * (1/math.cos(math.pi/2.0 - elt) - 1.0)

class ClearSkyModel():
    
    '''
    An implementation of Ineichen's 2008 Solis Clear Sky Model
    
    Usage:
    lat = 44.0
    lon = -80.0
    elev = 306
    tmp = 6
    
    loc = Location(lat,lon,elev)
    loc.setDST(3,12,11,5)
    loc.setTimeZoneOffsetFromGMT(-6)
    
    csm = ClearSkyModel(loc)
    
    now = datetime.now()
    
    
    for hour in range(5, 20):
        for minute in range(0, 60, 15):
            nexttime = datetime(now.year, now.month, now.day, hour, minute, 0)
            # You must re-initialze the time before each call, if no time is passed for the initialize
            # call, then the current runtime is used
            csm.initialize(nexttime)
            print(csm.getClearSky())           
            
    '''
    
    def __init__(self, location):
        self.location = location
        self.spc = SolarPositionCalculator(self.location)
        self.data = None
        
    def setTime(self, time):
        self.time = time
    
    # Get rid of this, write code to use setTime, setAODW instead
    # and load AOD and W from a separate class
    def initialize(self, time=None):
        if time:
            self.time = time
        else:
            self.time = datetime.now()
        aodfilename = 'aod_' + self.time.strftime('%Y%m%d') + '.xlsx'
        if self.data is None:            
            if os.path.isfile(aodfilename):
                self.loadData(aodfilename)
            else:
                self.callData(self.time)
            self.interpolate()
            
    def setAODW(self, aod, prw):
        self.aod = aod
        self.prw = prw
        
    def callData(self, time=None):
        year = time.year
        month = time.month
        day = time.day
        if not time:
            time = datetime.now()
        print("No stored data, so calling aeronet and storing data for {}".format(time))
        aodfilename = 'aod_' + time.strftime('%Y%m%d') + '.xlsx'
        url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?if_no_html=1&year={}&month={}&day={}&year2={}&month2={}&day2={}&AOD10=1&AVG=20".format(year,month,day,year,month,day)
        r = requests.get(url)
        self.data = pd.read_csv(url,skiprows=5)
        self.data = self.data[['AERONET_Site', 'AOD_675nm', 'Precipitable_Water(cm)', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'Site_Elevation(m)']]
        self.data.columns = ['Site','AOD','W','Lat','Lon','Elev']
        self.data.to_excel(aodfilename)
        
    def loadData(self, aodfilename):
        self.data = pd.read_excel(aodfilename)
        
    def interpolate(self):
        lats = self.data['Lat'].tolist()
        lons = self.data['Lon'].tolist()
        aods = self.data['AOD'].tolist()
        prws = self.data['W'].tolist()
        data = zip(lats,lons,aods,prws)
        eastings = []
        northings = []
        aoddata = []
        prwdata = []
        # This next variable sets how much of the data around the interplation point is included in the interpolation
        # Setting this too small risks having the point fall outside of a convex covering of the aod data
        # which causes the interplation to fail. 
        limit = 90.0
        for datum in data:
            if (abs(datum[1] - self.location.lon) < limit) and (abs(datum[0] - self.location.lat) < limit):
                if ((datum[2] != -999.0) and (datum[3] != -999.0)):
                    easting, northing = self.location.projectToUTM(datum[0], datum[1])
                    eastings.append(easting)
                    northings.append(northing)
                    aoddata.append(datum[2])
                    prwdata.append(datum[3])
        # Uncomment for trouble-shooting
        # print("The eastings and northings of the point to interpolate to is ({}, {})".format(self.location.easting, self.location.northing))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.scatter(np.asarray(eastings), np.asarray(northings), np.asarray(aoddata))
        # plt.show()
        self.aod = griddata(eastings, northings, aoddata, self.location.easting, self.location.northing, interp='linear')
        self.prw = griddata(eastings, northings, prwdata, self.location.easting, self.location.northing, interp='linear')
        
    def getClearSky(self):
        # Returns elevation (in radians), azimuth (in radians), DNI, GHI, DIF (W/m2)
        el, az = self.spc.getGeometry(self.time)
        if el < 0.0:
            return (el, az, 0.0, 0.0, 0.0)
        aod = self.aod
        w = self.prw
        # Fix this next bit, right now we aren't checking for pressure since its effect is minor anyway
        # We need a good API data source for the pressure
        p0 = 101325.0
        p = p0
        # The solar constant
        I0 = 1361.0
        I00 = 1.08 * w**0.0051
        I01 = 0.97 * w**0.032
        I02 = 0.12 * w**0.56
        I0p = I0 * (I02 * aod**2.0 + I01 * aod + I00 + 0.071*math.log(p/p0))
        # Beam Coefficient Values
        tb0 = 0.33 + 0.045*math.log(w) + 0.0096*(math.log(w))**2.0
        tb1 = 1.82 + 0.056*math.log(w) + 0.0071*(math.log(w))**2.0
        tbp = 0.0089*w + 0.13
        b1 = 0.00925*aod**2.0 + 0.0148*aod - 0.0172
        b0 = -0.7565*aod**2.0 + 0.5057*aod + 0.4457
        b = b1 * math.log(w) + b0
        tau_b = tb1*aod + tb0 + tbp*math.log(p/p0)
        #print("Tau_b is {}".format(tau_b))
        # Global Coefficient Values
        tg0 = 0.27 + 0.043*math.log(w) + 0.0090*(math.log(w))**2.0
        tg1 = 1.24 + 0.047*math.log(w) + 0.0061*(math.log(w))**2.0
        tgp = 0.0079*w + 0.1
        g = -0.0147*math.log(w) - 0.3079*aod**2.0 + 0.2846*aod + 0.3798
        tau_g = tg1*aod + tg0 + tgp*math.log(p/p0)
        # Diffuse Coefficient Values
        if aod < 0.05:
            td0 = 0.0042*w + 3.12
            td1 = 0.092*w - 8.86
            td2 = -0.23*w + 74.8
            td3 = -3.11*w + 79.4
            td4 = 86.0*w - 13800.0
            tdp = -0.83*(1.0 + aod)**(-17.2)
        else:
            td0 = 0.0057*w + 2.94
            td1 = 0.0554*w - 5.71
            td2 = -0.134*w + 15.5
            td3 = 0.27*w - 20.7
            td4 = -0.21*w - 11.6
            tdp = -0.71*(1.0 + aod)**(-15.0)
        dp = 1.0/(18.0 + 152.0*aod)
        d = -0.337*aod**2.0 + 0.63*aod + 0.116 + dp*math.log(p/p0)
        tau_d = td4*aod**4.0 + td3*aod**3.0 + td2*aod**2.0 + td1*aod + td0 + tdp*math.log(p/p0)
        Ibeam = I0p*math.exp(-1.0*tau_b/(math.sin(el)**b))
        if(np.isnan(Ibeam)):
            Ibeam = 0
        Iglob = I0p*math.exp(-1.0*tau_g/(math.sin(el)**g))*math.sin(el)
        if(np.isnan(Iglob)):
            Iglob = 0
        Idiff = I0p*math.exp(-1.0*tau_d/(math.sin(el)**d))
        if(np.isnan(Idiff)):
            Idiff = 0
        return (el, az, Ibeam, Iglob, Idiff)

class ClearSkyModel2():
    
    '''
    An implementation of Ineichen's 2008 Solis Clear Sky Model
    
    Usage:
    lat = 44.0
    lon = -80.0
    elev = 306
    tmp = 6
    
    loc = Location(lat,lon,elev)
    loc.setDST(3,12,11,5)
    loc.setTimeZoneOffsetFromGMT(-6)
    
    csm = ClearSkyModel(loc)
    
    now = datetime.now()
    
    
    for hour in range(5, 20):
        for minute in range(0, 60, 15):
            nexttime = datetime(now.year, now.month, now.day, hour, minute, 0)
            # You must re-initialze the time before each call, if no time is passed for the initialize
            # call, then the current runtime is used
            csm.initialize(nexttime)
            print(csm.getClearSky())           
            
    '''
    
    def __init__(self, location):
        self.location = location
        self.spc = SolarPositionCalculator2(self.location)
        self.data = None
        
    def setTime(self, time):
        self.time = time
    
    # Get rid of this, write code to use setTime, setAODW instead
    # and load AOD and W from a separate class
    def initialize(self, time=None):
        if time:
            self.time = time
        else:
            self.time = datetime.now()
        aodfilename = 'aod_' + self.time.strftime('%Y%m%d') + '.xlsx'
        if self.data is None:            
            if os.path.isfile(aodfilename):
                self.loadData(aodfilename)
            else:
                self.callData(self.time)
            self.interpolate()
            
    def setAODW(self, aod, prw):
        self.aod = aod
        self.prw = prw
        
    def callData(self, time=None):
        year = time.year
        month = time.month
        day = time.day
        if not time:
            time = datetime.now()
        print("No stored data, so calling aeronet and storing data for {}".format(time))
        aodfilename = 'aod_' + time.strftime('%Y%m%d') + '.xlsx'
        url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?if_no_html=1&year={}&month={}&day={}&year2={}&month2={}&day2={}&AOD10=1&AVG=20".format(year,month,day,year,month,day)
        r = requests.get(url)
        self.data = pd.read_csv(url,skiprows=5)
        self.data = self.data[['AERONET_Site', 'AOD_675nm', 'Precipitable_Water(cm)', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'Site_Elevation(m)']]
        self.data.columns = ['Site','AOD','W','Lat','Lon','Elev']
        self.data.to_excel(aodfilename)
        
    def loadData(self, aodfilename):
        self.data = pd.read_excel(aodfilename)
        
    def interpolate(self):
        lats = self.data['Lat'].tolist()
        lons = self.data['Lon'].tolist()
        aods = self.data['AOD'].tolist()
        prws = self.data['W'].tolist()
        data = zip(lats,lons,aods,prws)
        eastings = []
        northings = []
        aoddata = []
        prwdata = []
        # This next variable sets how much of the data around the interplation point is included in the interpolation
        # Setting this too small risks having the point fall outside of a convex covering of the aod data
        # which causes the interplation to fail. 
        limit = 90.0
        for datum in data:
            if (abs(datum[1] - self.location.lon) < limit) and (abs(datum[0] - self.location.lat) < limit):
                if ((datum[2] != -999.0) and (datum[3] != -999.0)):
                    easting, northing = self.location.projectToUTM(datum[0], datum[1])
                    eastings.append(easting)
                    northings.append(northing)
                    aoddata.append(datum[2])
                    prwdata.append(datum[3])
        # Uncomment for trouble-shooting
        # print("The eastings and northings of the point to interpolate to is ({}, {})".format(self.location.easting, self.location.northing))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.scatter(np.asarray(eastings), np.asarray(northings), np.asarray(aoddata))
        # plt.show()
        self.aod = griddata(eastings, northings, aoddata, self.location.easting, self.location.northing, interp='linear')
        self.prw = griddata(eastings, northings, prwdata, self.location.easting, self.location.northing, interp='linear')
        
    def getClearSky(self):
        # Returns elevation (in radians), azimuth (in radians), DNI, GHI, DIF (W/m2)
        el, az = self.spc.getGeometry(self.time)
        if el < 0.0:
            return (el, az, 0.0, 0.0, 0.0)
        aod = self.aod
        w = self.prw
        # Fix this next bit, right now we aren't checking for pressure since its effect is minor anyway
        # We need a good API data source for the pressure
        p0 = 101325.0
        p = p0
        # The solar constant
        I0 = 1361.0
        I00 = 1.08 * w**0.0051
        I01 = 0.97 * w**0.032
        I02 = 0.12 * w**0.56
        I0p = I0 * (I02 * aod**2.0 + I01 * aod + I00 + 0.071*math.log(p/p0))
        # Beam Coefficient Values
        tb0 = 0.33 + 0.045*math.log(w) + 0.0096*(math.log(w))**2.0
        tb1 = 1.82 + 0.056*math.log(w) + 0.0071*(math.log(w))**2.0
        tbp = 0.0089*w + 0.13
        b1 = 0.00925*aod**2.0 + 0.0148*aod - 0.0172
        b0 = -0.7565*aod**2.0 + 0.5057*aod + 0.4457
        b = b1 * math.log(w) + b0
        tau_b = tb1*aod + tb0 + tbp*math.log(p/p0)
        #print("Tau_b is {}".format(tau_b))
        # Global Coefficient Values
        tg0 = 0.27 + 0.043*math.log(w) + 0.0090*(math.log(w))**2.0
        tg1 = 1.24 + 0.047*math.log(w) + 0.0061*(math.log(w))**2.0
        tgp = 0.0079*w + 0.1
        g = -0.0147*math.log(w) - 0.3079*aod**2.0 + 0.2846*aod + 0.3798
        tau_g = tg1*aod + tg0 + tgp*math.log(p/p0)
        # Diffuse Coefficient Values
        if aod < 0.05:
            td0 = 0.0042*w + 3.12
            td1 = 0.092*w - 8.86
            td2 = -0.23*w + 74.8
            td3 = -3.11*w + 79.4
            td4 = 86.0*w - 13800.0
            tdp = -0.83*(1.0 + aod)**(-17.2)
        else:
            td0 = 0.0057*w + 2.94
            td1 = 0.0554*w - 5.71
            td2 = -0.134*w + 15.5
            td3 = 0.27*w - 20.7
            td4 = -0.21*w - 11.6
            tdp = -0.71*(1.0 + aod)**(-15.0)
        dp = 1.0/(18.0 + 152.0*aod)
        d = -0.337*aod**2.0 + 0.63*aod + 0.116 + dp*math.log(p/p0)
        tau_d = td4*aod**4.0 + td3*aod**3.0 + td2*aod**2.0 + td1*aod + td0 + tdp*math.log(p/p0)
        Ibeam = I0p*math.exp(-1.0*tau_b/(math.sin(el)**b))
        if(np.isnan(Ibeam)):
            Ibeam = 0
        Iglob = I0p*math.exp(-1.0*tau_g/(math.sin(el)**g))*math.sin(el)
        if(np.isnan(Iglob)):
            Iglob = 0
        Idiff = I0p*math.exp(-1.0*tau_d/(math.sin(el)**d))
        if(np.isnan(Idiff)):
            Idiff = 0
        return (el, az, Ibeam, Iglob, Idiff)

class RadianceCalculation(Model):
    
    '''
    Creates a Radiance geometry file for a tracker then calls Radiance to calculate rradiances at the front and back
    panel submodules.
    
    Usage:
    
    directory = '/home/geoffrey/Backup/CurrentProjects/Aruna/Research/Radiance/Calculations/'
    fname = 'panels'
    rc = RadianceCalculation()
    rc.createTrackerGeometry(2.0, 992.0, 10.0, 1956.0, 40.0, 2.0, 2.0, 500.0, 25.0, 5, 5000.0, Model.ONE_PORTRAIT, directory, fname)
    irrads = rc.getIrradianceAcrossPanel(60, 0, 600, 200, 0.251, directory, fname)
    
    '''
    
    
    def createTrackerGeometry(self, pw, wp, pl, lp, hp, ws, ls, ha, theta, rows, pitch, configuration, directory, fname):
        pitch_mm = 1000.0 * pitch
        # Multiply angle by -1 to match convention that positive panel angles occur in morning
        theta = -1.0 * theta
        self.geometry = [pw, wp, pl, lp, hp, ws, ls, ha, theta, rows, pitch]        
        w  = pw * wp + (pw - 1.0) * ws       # wp = width of panels, pw = number of panels in width, ws = spacing between panels in width
        l  = pl * lp + (pl - 1.0) * ls       # lp = length of panels, pl = number of panels in length, ls = spacing between panels in length, ha = tracker axis height
        h  = hp                              # theta = panel angle from horizontal, pitch = distance between panel rows axes 
        r  = 0.5 * (w**2.0 + h**2.0)**0.5    
        a  = math.atan2(h, w)
        theta = theta * 3.1415925 / 180.0
        a1 = theta + a                       # Points 1 and 4 correspond to the upper surface
        a2 = theta - a                       # Points 2 and 3 correspond to the lower surface
        a3 = theta + math.pi - a
        a4 = theta + math.pi + a
        x1 = r * math.cos(a1)
        z1 = r * math.sin(a1) + ha           # ha is the height of the tracking axis off the ground
        x2 = r * math.cos(a2)
        z2 = r * math.sin(a2) + ha
        x3 = r * math.cos(a3)
        z3 = r * math.sin(a3) + ha
        x4 = r * math.cos(a4)
        z4 = r * math.sin(a4) + ha
        mts = float(rows // 2) * pitch * 1000.0       # The shift x-wards to get to the middle tracker
        print("MTS is {}".format(mts))
        self.vertex_coords = (x1 + mts, x2 + mts, x3 + mts, x4 + mts, z1, z2, z3, z4, l/2.0)
        self.normals = ((-1.0 * math.sin(theta), 0.0, math.cos(theta)), (math.sin(theta), 0.0, -1.0 * math.cos(theta)))
        p3 = [x1, 0, z1]
        p4 = [x2, 0, z2]
        p1 = [x3, 0, z3]
        p2 = [x4, 0, z4]
        p5 = [x1, l, z1]
        p8 = [x2, l, z2]
        p7 = [x3, l, z3]
        p6 = [x4, l, z4]
        vertexes = [p1, p2, p3, p4, p5, p6, p7, p8]
        faces = [ [1, 2, 3], [2, 4, 3], [5, 6, 7], [5, 8, 6],\
                  [4, 8, 5], [3, 4, 5], [7, 6, 2], [7, 2, 1],\
                  [4, 2, 6], [8, 4, 6], [7, 1, 3], [7, 3, 5] ]
        x1, x2, x3, x4, z1, z2, z3, z4, y = self.vertex_coords
        print("X1 is {}".format(x1))
        xtu = ztu = xtm = ztm = xtl = ztl = xbu = zu = xbm = zbm = xbl = zbl = 0
        xtn, ytn, ztn = self.normals[0]
        xbn, ybn, zbn = self.normals[1]
        if configuration == self.ONE_PORTRAIT:
            # Need to have the measurement locations a little offset from the panels, otherwise Radiance fails
            offset = 40.0
            xtu = (5.0 * x1 + 1.0 * x3) / 6.0 + offset * self.normals[0][0]
            ztu = (5.0 * z1 + 1.0 * z3) / 6.0 + offset * self.normals[0][2] 
            xtm = (1.0 * x1 + 1.0 * x3) / 2.0 + offset * self.normals[0][0]
            ztm = (1.0 * z1 + 1.0 * z3) / 2.0 + offset * self.normals[0][2] 
            xtl = (1.0 * x1 + 5.0 * x3) / 6.0 + offset * self.normals[0][0]
            ztl = (1.0 * z1 + 5.0 * z3) / 6.0 + offset * self.normals[0][2] 
            xbu = (5.0 * x2 + 1.0 * x4) / 6.0 + offset * self.normals[1][0]
            zbu = (5.0 * z2 + 1.0 * z4) / 6.0 + offset * self.normals[1][2] 
            xbm = (1.0 * x2 + 1.0 * x4) / 2.0 + offset * self.normals[1][0]
            zbm = (1.0 * z2 + 1.0 * z4) / 2.0 + offset * self.normals[1][2]  
            xbl = (1.0 * x2 + 5.0 * x4) / 6.0 + offset * self.normals[1][0]
            zbl = (1.0 * z2 + 5.0 * z4) / 6.0 + offset * self.normals[1][2] 
        self.irrad_locs = [[[xtu, y, ztu, xtn, ytn, ztn], [xtm, y, ztm, xtn, ytn, ztn], [xtl, y, ztl, xtn, ytn, ztn]], [[xbu, y, zbu, xbn, ybn, zbn], [xbm, y, zbm, xbn, ybn, zbn], [xbl, y, zbl, xbn, ybn, zbn]]]
        outfile =  open(directory + fname + '.obj', 'w')
        for row in range(rows):
            for vertex in vertexes:
                outfile.write('v ' + str(vertex[0] + pitch_mm * row)  + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')
        for row in range(rows):
            for face in faces:
                outfile.write('f ' + str(face[0] +  8 * row) + ' ' + str(face[1] + 8 * row) + ' ' + str(face[2] + 8 * row) + '\n')
        outfile.close()
        os.system(Model.RADBINS + 'obj2rad ' + directory + fname + '.obj > ' + directory + fname + '.rad')
        
    def createNearShadingGeometry(self, directory, shade_elements = None):
        outfile = open(directory + 'nearShading.rad', 'w')
        if shade_elements:
            for element in shade_elements:
                outfile.write(element + '\n')
        else:
            outfile.write('')
        outfile.close()
        
    def getIrradianceAcrossPanel(self, sun_angle, sun_azimuth, dni, dhi, albedo, directory, fname, animate, file_name, calculate, shadings):
        # Radiance uses a convention Azimuth defined west of south so subtract 180 degrees from the azimuth to give the correct value
        sun_azimuth = sun_azimuth - 180.0
        print("Elevation: {}, Azimth: {}".format(sun_angle, sun_azimuth))
        os.putenv('RAYPATH', '/home/geoffrey/Backup/Install/radiance-5.2.a46558bb5f-Linux/usr/local/radiance/lib/')
        os.system(Model.RADBINS + 'gendaylit -ang ' + str(sun_angle) + ' ' + str(sun_azimuth) + ' -W ' + str(dni) + ' ' + str(dhi) + ' -g ' + str(albedo) + ' -O 1 > ' + directory + 'sun.rad')
        if shadings:
            os.system(Model.RADBINS + 'oconv ' + directory + 'sun.rad ' + directory + 'sky.rad ' + directory + 'ground.rad '  + directory + 'panelcolour.rad ' + directory + fname + '.rad ' + directory + 'nearShading.rad > ' + directory + fname + '.oct')
        else:
            os.system(Model.RADBINS + 'oconv ' + directory + 'sun.rad ' + directory + 'sky.rad ' + directory + 'ground.rad '  + directory + 'panelcolour.rad ' + directory + fname + '.rad > ' + directory + fname + '.oct')
        if animate:
            # os.system(Model.RADBINS + 'rpict -vp 0 0 40000 -vd 0.01 0.01 -1.0 -av 0 0 0 -ab 8 -ad 128 -aa 0.1 -ar 512 ' + directory + 'panels.oct > ' + directory + 'panels.hdr')
            # os.system(Model.RADBINS + 'rpict -vp 23800 -4000 3000 -vd -0.25 4 -0.25 -av 0 0 0 -ab 4 -ad 384 -aa 0.1 -ar 512 ' + directory + 'panels.oct > ' + directory + 'panels.hdr')
            # Use this for quick runs The pictures generated will be poor but the calculations for mismatch are not calculated by this next line
            # os.system(Model.RADBINS + 'rpict -vp 23800 -4000 3000 -vd -0.25 4 -0.25 -av 0 0 0 -ab 2 -ad 12 -aa 10 -ar 64 ' + directory + 'panels.oct > ' + directory + 'panels.hdr')
            os.system(Model.RADBINS + 'rpict -vp -45000 25000 45000 -vd 1 0 -1 -av 0 0 0 -ab 2 -ad 384 -aa 0.1 -ar 512 ' + directory + 'panels.oct > ' + directory + 'panels.hdr')
            os.system(Model.RADBINS + 'ra_tiff -e -6 ' + directory + 'panels.hdr ' + directory + file_name + '.tif')
        irrads = []
        if calculate:
            for surface in range(2):
                for position in range(3):
                    # Get the 3 coordinates as a string for three locations near the center of the tracker - given as irrad_locs. Get the coord strings 1m either side of the center of the tracker to average together.
                    # This averageing of the 5 points reduces errors in the result
                    coordstring = str(self.irrad_locs[surface][position][0]) + ' ' + str(self.irrad_locs[surface][position][1]) + ' ' + str(self.irrad_locs[surface][position][2] + 1.0) + ' ' + str(self.irrad_locs[surface][position][3]) + ' ' + str(self.irrad_locs[surface][position][4]) + ' ' + str(self.irrad_locs[surface][position][5])  
                    return_val = out("echo '" + coordstring + "' | " + Model.RADBINS + "rtrace -I+ -n 4 -ab 6 -ad 512 -aa 0.1 -ar 512 -h " + directory + fname + ".oct | " + Model.RADBINS + "rcalc -e '$1=0.265*$1+0.670*$2+0.065*$3'")
                    irrads.append(float(return_val[:-2]))
        return irrads


        
        
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
        

'''

easting, northing = loc.projectToUTM(43.214089, -79.711568)

clm = ClearSkyModel(loc)
time = datetime(2018,3,31,12,0,0)
clm.initialize(time)
print(clm.aod, clm.prw)
'''



'''
time = [2017,6,21,21,10,0]
lat = 43.5
lon = -80.5
elev = 86
tmp = -10

loc = Location(lat,lon,elev)
loc.setDST(3,12,11,5)
loc.setTimeZoneOffsetFromGMT(-5)
dt = datetime(time[0],time[1],time[2],time[3],time[4],time[5])

iam = IAMModel(0.05,loc,0)
print(iam.getIAM(dt))

iam = IAMModel(0.05,loc,21)
print(iam.getIAM(dt))

'''


'''
class ShadingModel():
    def __init__(self, configuration):
        self.configuration = configuration
'''

'''
class ClearSkyModel():

    def __init__(self, location):
        self.location = location
        self.spc = SolarPositionCalculator(location)
        self.solarConstant = 1367.0

    def getPredictedIrradiation(self, time, temp, at_p=0.0, humidity=0.90, turbidity=0.5):
        day = time.timetuple().tm_yday
        altitude, azimuth = self.spc.getGeometry(time)
        # This next bit calculates the inverse relative distance factor between the earth and the sun
        sol_r = 1.0 / (1.0 - 9.464e-4 * math.sin(day) - 0.01671 * math.cos(day) \
                       - 1.489e-4 * math.cos(2.0 * day) - 2.917e-5 * math.sin(3.0 * day) \
                       - 3.438e-4 * math.cos(4.0 * day)) ** 2.0
        # Retrieve the solar declination already calculated by the SolarPositionCalculator
        sol_d = self.spc.declination
        # Calculate the barometric pressure at the site - uses a measured value if it was input in the method call
        if at_p == 0.0:
            at_p = math.pow(((288.0 - (0.0065 * (altitude - 0))) / 288.0), (9.80665 / (0.0065 * 287.0))) * 101.325
        # Calculate the estimated air vapour pressure
        vp = (0.61121 * math.exp((17.502 * temp) / (240.97 + temp))) * (humidity / 100)
        # Now, calculate the extra-terrestrial radiation
        etr = (self.solarConstant * sol_r) * (math.cos(math.pi / 2.0 - altitude))
        # Find the precipitable water in the atmosphere
        pw = ((0.14 * vp) * at_p) + 2.1
        # Next, find the clearness index for the direct irradiation component
        ci = 0.98 * (math.exp(((-0.00146 * at_p) / (turbidity * (math.sin(altitude)))) - (0.075 * (math.pow((pw / (math.sin(altitude))), 0.4)))))
        # From the clearness index, get the transmissivity index for the diffuse irradiation component
        if (ci > 0.15):
            ti = 0.35 - (0.36 * ci)
        else:
            ti = 0.18 + (0.82 * ci)
        # Finally, get the transmitted irradation
        irrad = (ci + ti) * etr
        # Return the direct, diffuse and global components
        return (ci * etr, ti * etr, irrad)


'''
