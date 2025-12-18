from core import EnvironmentGenerator, Model, StringPool, BFModule
from utilities import Location, SolarPositionCalculator, SolarPositionCalculator2, IAMModel, ClearSkyModel
from utilities import SolarGeometry, SeparationTranspositionModel, RadianceCalculation
from utilities import NewtonsMethod

from datetime import datetime
from collections import defaultdict
import math
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import bisect

class Error(Exception):
	"""Base class for Exceptions in this module"""
	pass

class ConfigurationError(Error):
	"""
	Attributes:
		expression -- input expression which resulted in the error
		message -- explanation of the error
	"""

	def __init__(self,  message, configuration):
		self.message = message
		self.config = configuration

class Table(Model):

	'''
	The model assumes that an integral number of strings will fit on each Table, so
	num_strings = width * height = N * string_length where N is an integer.
	A StringPool is created with a pool size of num_strings in the method addPanels which creates the panels
	using the parameters passed to it.
	Modules (groups of 3 panels) are then pulled from this StringPool and placed on the Table in a manner consistent
	with the parameters passed to the method placePanels.
	After this, shadingFactors are set, indicating inter-table shading, by running the setShadingFactors method which
	takes a datetime object as a parameter, calculating shading by determining the sun's position for that time.
	
	Usage:
	
	Create a solar farm, and for a particular day, see the production over that day using a clear sky model.

	lat = 43.55
	lon = -80.5
	elev = 125
	tmp = -10
	
	simulation_year = 0
	
	loc = Location(lat,lon,elev)
	loc.setDST(3,12,11,5)
	loc.setTimeZoneOffsetFromGMT(-6)
	
	panel_params = [24.0, 15.10, 9.26, 0.1167, 900.0, 1.956, 0.992, "Canadian Solar CS6U320P", 106.667, 3, 0.0046, -0.152, 1.026]
	inverter_params = ['Sungrow SG60K-U', 30.0, 4, 0.05, 350.0, 850.0]   # Inverter power, number of MPPTS, min power per MPPT, minimum MPPT voltage, maximum MPPT voltage
	system_design_params = [1.2, False, None]
	panel_degradation_params = [0.998, 0.0002, 0.993, 0.0002, 1.01, 0.1, 0.975, 0.015] # These ratios are rough estimates from the paper - Reis et al, 2002
	panel_manufacturing_params = [1.0, 0.0072, 1.0, 0.0134]  # These ratios are rough estimates from the paper - Reis et al, 2002
	panel_manufacturing_tolerances = [0.0, 5.0 / 315.0 * 100.0]  # Typical manufacturing tolerance 315 W (-0W/+5W)
	eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 2.0, 1100.0, 100.0, 0.02, 3.0)
	eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
	
	tilt = 25.0
	
	table = Table(loc,9, 4, tilt, 180.0, 8.5, 18)
	table.addPanels(eg, panel_params, panel_manufacturing_params, panel_manufacturing_tolerances, panel_degradation_params, simulation_year)
	table.placePanels("l","w")
	
	mppt = MPPT(inverter_params, system_design_params, table)
	mppt.assignPanels()
	
	dt = datetime.now()
	
	csm = ClearSkyModel(loc)
	
	sg = SolarGeometry()
	
	stm = SeparationTranspositionModel(loc)
	
	irrads = []
	voltages = []
	powers = []
	times = []
	
	for hour in range(6, 20):
		for minute in range(0, 60, 5):
			dt = datetime(dt.year, dt.month, dt.day, hour, minute, 0)
			csm.initialize(dt)
			el, az, dni, ghi, dif = csm.getClearSky()
			altr, azir = sg.getGeometryInTiltPlane(el, az, tilt)
			irr = stm.transpositionModel(dni, dif, el, tilt)
			eg.setIm(irr)
			istd = random.random()*irr
			eg.setIs(istd)
			mppt.generateIVCurve(simulation_year, dt)
			mppt_data = mppt.getMPP()
			mppt.plotPowerCurve(mppt_data, dt)
			times.append(dt)
			irrads.append(irr)
			voltages.append(mppt_data[0])
			powers.append(mppt_data[2])
			
	plt.plot(powers)
	plt.show()
		
	

	'''

	def __init__(self, location, width, height, tilt, azimuth, pitch, string_length):
		self.spc = SolarPositionCalculator(location)
		self.iam = IAMModel(0.05,location,tilt)
		self.width = width
		self.height = height
		self.tilt = tilt
		self.azimuth = azimuth
		self.pitch = pitch
		self.string_length = string_length
		self.strings = defaultdict(list)
		self.table_strings = []
		self.module_rows = defaultdict(list)
		if not width * height % string_length == 0:
			raise ConfigurationError("An integral number of strings does not fit the table!", [width, height, string_length])
		else:
			self.num_strings = int(width * height / string_length)

	def __iter__(self):
		self.string_keys = list(self.strings.keys())   # In P3, the keys method does not return a list but a dict_keys object
		self.key_index = 0                             # Here, we just reset the iterator, setting the index to 0 and get the list of keys
		return self

	def __next__(self):
		if self.key_index < len(self.string_keys):
			next_string = self.strings[self.string_keys[self.key_index]]
			self.key_index += 1
			return next_string
		else:
			return StopIteration

	def set_loop(self):                               # This and the following method set up an infinite repeating loop, cycling through strings
		self.loop_index = 0                           # This is used for filling in MPPTs while re-using Table strings whose creation is expensive

	def loop(self):
		next_string = self.table_strings[self.loop_index]
		self.loop_index = (self.loop_index + 1) % self.num_strings
		return next_string

	def resetEnvironment(self):
		for module in self.stringPool:
			module.resetEnvironment()

	def addPanels(self, eg, params, manufacturingParams, tolerances, degradationParams, year):

		'''
		This method creates the panels and adds them to the StringPool for this class
		The StringPool is responsible for the complete management of the creation and time-degradation
		of the modules that it creates, using the core classes in the 'core' module
		'''

		self.params = params
		self.manufacturing_params = manufacturingParams
		self.degradation_params = degradationParams
		self.tolerances = tolerances
		self.stringPool = StringPool(eg, params, manufacturingParams, tolerances, degradationParams,
										  self.string_length * params[self.SUBMODULES], self.num_strings)
		self.stringPool.generatePool(year)
		
	def initializeShadingFactors(self):
		for string_index, string in enumerate(self.stringPool):
			for module_index, module in enumerate(string):
				module.initializeShadingFactors()

	def placePanels(self, module_orientation = "l", stringing_direction = "w"):

		'''
		This method is responsible to placing the modules in the table, following the orientation and stringing direction
		assigned for the panels. It is required for this method at present, that an integral number of strings 4
		exactly fill the modules of the table, so self.height * self.width = N * string_length, where N is an integer.
		'''

		self.module_orientation = module_orientation
		self.stringing_direction = stringing_direction
		module_width = self.stringPool.params[self.stringPool.W]
		module_height = self.stringPool.params[self.stringPool.H]
		if module_orientation not in ["l", "p"]:
			raise ConfigurationError("Panels must be in landscape (l) or portrait (p) orientation", module_orientation)
		if module_orientation == 'l':
			self.w = self.height * module_width
			self.shading_unit = module_width
		else:
			self.w = self.height * module_height
			self.shading_unit = module_height
		if stringing_direction not in ["w", "h"]:
			raise ConfigurationError("Panel stringing direction must be width-wise (w) or height-wise (h)", stringing_direction)
		for string_index, string in enumerate(self.stringPool):
			self.table_strings.append(string)
			for module_index, module in enumerate(string):
				self.strings[string_index].append(module)
				index = string_index * self.string_length + module_index
				dir_length = self.width
				if stringing_direction == "h":
					dir_length = self.height
				n = int(index/dir_length)
				if n % 2 == 0:
					m = index % dir_length
				else:
					m = (dir_length - 1) - index % dir_length
				if stringing_direction == "w":
					module.setConfiguration(self, module_orientation, m, n)
				else:
					module.setConfiguration(self, module_orientation, n, m)
				self.module_rows[n].append(module)

	def getPOAIrradiance(self,time):

		'''
		This method determines the POA irradiance for the table, calculating the effects of shading and incidence
		angle modifiers on the irradiance. It uses the SolarPositionCalculator to find out where the sun is at the
		input time, then determines the extent of shading on the panels, then uses the IAM Model to determine how
		much to reduce the irradiance due to light reflection off the panels.
		'''

		#First, get the IAM for the table at the particular time
		iam_factor = self.iam.getIAM(time)
		# Re-initialize the shading factors here
		self.initializeShadingFactors()
		#Next, find the extent of the shading
		solar_elevation, solar_azimuth = self.spc.getGeometryAsAngles(time)
		relative_azimuth = solar_azimuth - self.azimuth
		relative_elevation = math.atan2(math.tan(solar_elevation * self.dtr),math.cos(relative_azimuth * self.dtr))
		shading_length = (1/math.sin(self.tilt * self.dtr))* \
						 (((self.w * math.sin(self.tilt * self.dtr)/(math.tan(relative_elevation))) -
						 self.pitch + self.w*math.cos(self.tilt * self.dtr))
						 /(1/math.tan(relative_elevation) + 1/math.tan(self.tilt * self.dtr)))
		shading_length = max(shading_length, 0)
		units_shaded = shading_length / self.shading_unit
		fully_shaded = int(units_shaded)
		fraction_shaded = units_shaded - fully_shaded
		#print("The number of units shaded is {} and the fraction of the last row shaded is {}".format(fully_shaded, fraction_shaded))
		if self.module_orientation == 'l':
			if fully_shaded:
				for row in range(fully_shaded):
					for module in self.module_rows[row]:
						for panel in module:
							panel.setShadingFactor(0.0)
							#print("Shading factor is {}".format(panel.getShadingFactor()))
			for module in self.module_rows[fully_shaded]:
				panels_shaded = fraction_shaded * module.getNumberOfPanelsPerModule()
				panels_fully_shaded = int(panels_shaded)
				panels_fraction_shaded = panels_shaded - panels_fully_shaded
				for index, panel in enumerate(module):
					if index < panels_fully_shaded:
						panel.setShadingFactor(0.0)
						#print("Shading factor is {}".format(panel.getShadingFactor()))
					elif index == panels_fully_shaded:
						# At this point, we have the fraction of shading, but we must also apply our iam_factor
						panel.setShadingFactor((1.0 - panels_fraction_shaded) * iam_factor)
						#print("Shading factor is {}".format(panel.getShadingFactor()))
					else:
						break
		elif self.module_orientation == 'p':
			if fully_shaded:
				for row in range(fully_shaded):
					for module in self.module_rows[row]:
						for panel in module:
							panel.setShadingFactor(0.0)
							#print("Shading factor is {}".format(panel.getShadingFactor()))
			for module in self.module_rows[fully_shaded]:
				for panel in module:
					# At this point, we have the fraction of shading, but we must also apply our iam_factor
					panel.setShadingFactor((1.0 - panels_fraction_shaded) * iam_factor)
					#print("Shading factor is {}".format(panel.getShadingFactor()))

	def getIVCurve(self, year):
		for next_string in self.stringPool:
			next_string.getIVCurve(year)

class SAT2(Model):
	
	def __init__(self, location, width, tracker_phi, pitch, string_length):
		self.spc = SolarPositionCalculator2(location)
		# The panel width and the row pitch
		self.width = width
		self.pitch  = pitch
		# The phi angle for the tracker
		self.tracker_phi = tracker_phi * self.dtr
		# self.tracker_phi = 10000
		# Number of panels per string
		self.string_length = string_length
		# alpha_c is the angle of the sun's shadow on a vertcal plane running along an east-west axis
		self.f_alpha_c = lambda alpha_c: math.tan(alpha_c) - (self.width * math.cos(alpha_c)) / (self.pitch - self.width * math.sin(alpha_c))
		# acbt is the angle of the sun's shadow on a vertical plane running along an east-west axis when back-tracking initiates on level ground
		self.acbt = fsolve(self.f_alpha_c, 0.02)[0]

	def getBacktrackingAndPanelAngleFromBeta(self, beta):
		f_delta = lambda delta: math.tan(beta) - (self.width * math.sin(delta)) / (
					self.pitch - self.width * math.cos(delta))
		panel_angle = 0.0
		back_tracking = True
		if beta > 0:
			if beta < self.acbt:
				delta = fsolve(f_delta, 0.02)[0]
				panel_angle = delta
				if panel_angle > self.tracker_phi:
					panel_angle = self.tracker_phi
			else:
				back_tracking = False
				panel_angle = math.pi / 2.0 - beta
				if panel_angle > self.tracker_phi:
					panel_angle = self.tracker_phi
		else:
			if beta > math.pi - self.acbt:
				delta = fsolve(f_delta, 0.02)[0]
				panel_angle = delta
				if panel_angle < -1.0 * self.tracker_phi:
					panel_angle = -1.0 * self.tracker_phi
			else:
				back_tracking = False
				panel_angle = math.pi / 2.0 - beta
				if panel_angle < -1.0 * self.tracker_phi:
					panel_angle = -1.0 * self.tracker_phi

		return ( back_tracking, panel_angle )
		
	def getTrackerConfiguration(self,time):
		theta, phi = self.spc.getGeometry(time)
		day = False
		if theta > 0.0:
			day = True
		# alpha_c is the angle of the sun's shadow on a vertical plane running along an east-west axis
		alpha_c = math.atan2(math.tan(theta), math.sin(phi))
		#f_delta solves the equation tan(alpha_c) = (width * sin(delta)) / (pitch - width * cos(delta))
		# This equation is used to determine the panel angle during back-tracking, whhich is given by delta when f_delta = 0
		f_delta = lambda delta: math.tan(alpha_c) - (self.width * math.sin(delta))/(self.pitch - self.width * math.cos(delta))
		elevation = 0.0
		panel_angle = 0.0
		back_tracking = True
		# If the azimuth angle phi is less than 180 degrees then the time is before solar noon
		if phi * 180.0/math.pi < 180.0:
			if alpha_c < self.acbt:
				delta = fsolve(f_delta, 0.02)[0]
				panel_angle = delta
				if panel_angle > self.tracker_phi:
					panel_angle = self.tracker_phi
			else:
				back_tracking = False
				panel_angle = math.pi/2.0 - alpha_c
				if panel_angle > self.tracker_phi:
					panel_angle = self.tracker_phi
			beta = math.atan2( math.tan( theta ), math.sin( phi ) )
			h = math.cos( math.asin( math.cos( theta ) * math.cos( phi ) ) ) * math.sin( beta + panel_angle )
			l = ( math.cos( math.asin( math.cos( theta ) * math.cos( phi ) ) ) ** 2.0 * math.cos( beta + panel_angle ) ** 2.0 + ( math.cos( theta ) * math.cos( phi ) ) ** 2.0 ) ** 0.5
			elevation = math.atan2( h, l )
		# here, phi is greater than 1890 degrees so the time is after solar noon
		else:
			# the next condition determines if alpha_c is close enough to the west horizon to be in back-tracking mode
			if alpha_c >  math.pi - self.acbt:
				delta = fsolve(f_delta, 0.02)[0]
				panel_angle = delta
				if panel_angle < -1.0 * self.tracker_phi:
					panel_angle = -1.0 * self.tracker_phi
			else:
				back_tracking = False
				panel_angle = math.pi/2.0 - alpha_c
				if panel_angle < -1.0 * self.tracker_phi:
					panel_angle = -1.0 * self.tracker_phi
			beta = math.atan2( math.tan( theta ), math.sin( phi ) )
			h = math.cos( math.asin( math.cos( theta ) * math.cos( phi ) ) ) * math.sin( beta + panel_angle )
			l = ( math.cos( math.asin( math.cos( theta ) * math.cos( phi ) ) ) ** 2.0 * math.cos( beta + panel_angle ) ** 2.0 + ( math.cos( theta ) * math.cos( phi ) ) ** 2.0 ) ** 0.5
			elevation = math.atan2( h, l )
		return (day, back_tracking, elevation, panel_angle, theta, phi)

class SATSloped(Model):

	# This class is a model for a single axis tracker, which calcualtes the tracker configuration in a case where the ground is not level.
	# It assumes the worst case scennario, where it is assumed that the ground slopes upward towards the east of the tracker and the west,
	# so backtracking will have to compensate for ground slope in the morning and the afternoon. 
	
	def __init__(self, location, width, tracker_phi, pitch, ground_slope ):

		self.deg2rad = math.pi / 180.0
		self.rad2deg = 180.0 / math.pi

		self.spc = SolarPositionCalculator2(location)
		# The panel width and the row pitch
		self.width = width
		self.pitch  = pitch
		# The phi angle for the tracker
		self.tracker_phi = tracker_phi * self.dtr
		self.ground_slope = ground_slope * self.deg2rad
		# alpha_c is the angle of the sun's shadow on a vertcal plane running along an east-west axis
		self.f_alpha_c = lambda alpha_c: math.tan(alpha_c) - (self.width * math.cos(alpha_c)) / (self.pitch - self.width * math.sin(alpha_c))
		# acbt is the angle of the sun's shadow on a vertical plane running along an east-west axis when back-tracking initiates on level ground	
		self.acbt = fsolve(self.f_alpha_c, 0.02)[0]

		

	def getTrackerConfiguration(self, time):
		theta, phi = self.spc.getGeometry(time)
		day = False
		if theta > 0.0:
			day = True
		# alpha_c is the angle of the sun's shadow on a vertical plane running along an east-west axis
		alpha_c = math.atan2(math.tan(theta), math.sin(phi))
		#f_delta solves the equation tan(alpha_c) = (width * sin(delta)) / (pitch - width * cos(delta))
		# This equation is used to determine the panel angle during back-tracking, whhich is given by delta when f_delta = 0
		f_delta_morning = lambda delta: math.tan(alpha_c-self.ground_slope) - (self.width * math.sin(delta))/(self.pitch - self.width * math.cos(delta))
		f_delta_afternoon = lambda delta: math.tan(alpha_c+self.ground_slope) - (self.width * math.sin(delta))/(self.pitch - self.width * math.cos(delta))
		elevation = 0.0
		panel_angle = 0.0
		back_tracking = True
		# If the azimuth angle phi is less than 180 degrees then the time is before solar noon
		if phi * 180.0/math.pi < 180.0:
			if alpha_c < self.acbt + self.ground_slope:
				delta = fsolve(f_delta_morning, 0.02)[0]
				panel_angle = delta
				if panel_angle > self.tracker_phi:
					panel_angle = self.tracker_phi
				elif panel_angle < 0:
					panel_angle = 0
			else:
				back_tracking = False
				panel_angle = math.pi/2.0 - alpha_c
				if panel_angle > self.tracker_phi:
					panel_angle = self.tracker_phi			
		# here, phi is greater than 180 degrees so the time is after solar noon
		else:
			# the next condition determines if alpha_c is close enough to the west horizon to be in back-tracking mode
			if alpha_c >  math.pi - self.acbt - self.ground_slope:
				delta = fsolve(f_delta_afternoon, 0.02)[0]
				panel_angle = delta
				if panel_angle < -1.0 * self.tracker_phi:
					panel_angle = -1.0 * self.tracker_phi
				elif panel_angle > 0:
					panel_angle = 0
			else:
				back_tracking = False
				panel_angle = math.pi/2.0 - alpha_c
				if panel_angle < -1.0 * self.tracker_phi:
					panel_angle = -1.0 * self.tracker_phi
		perpendicularity = math.sin(panel_angle) * math.cos(theta) * math.sin(phi) + math.cos(panel_angle) * math.sin(theta)	
		return (day, back_tracking, panel_angle, theta, phi, perpendicularity)

class MPPT(Model):
	
	'''
	
	Usage:
	
	lat = 43.55
	lon = -80.5
	elev = 125
	tmp = -10
	
	simulation_year = 0
	
	loc = Location(lat,lon,elev)
	loc.setDST(3,12,11,5)
	loc.setTimeZoneOffsetFromGMT(-6)
	
	panel_params = [24.0, 15.10, 9.26, 0.1167, 900.0, 1.956, 0.992, "Canadian Solar CS6U320P", 106.667, 3, 0.0046, -0.152, 1.026]
	inverter_params = ['Sungrow SG60K-U', 30.0, 4, 0.05, 350.0, 850.0]   # Inverter power, number of MPPTS, min power per MPPT, minimum MPPT voltage, maximum MPPT voltage
	system_design_params = [1.2, False, None]
	panel_degradation_params = [0.998, 0.0002, 0.993, 0.0002, 1.01, 0.1, 0.975, 0.015] # These ratios are rough estimates from the paper - Reis et al, 2002
	panel_manufacturing_params = [1.0, 0.0072, 1.0, 0.0134]  # These ratios are rough estimates from the paper - Reis et al, 2002
	panel_manufacturing_tolerances = [0.0, 5.0 / 315.0 * 100.0]  # Typical manufacturing tolerance 315 W (-0W/+5W)
	eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 40.0, 1100.0, 800.0, 0.02, 3.0)
	eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
	
	table = Table(loc,9,4,25.0,180.0,8.5,18)
	table.addPanels(eg, panel_params, panel_manufacturing_params, panel_manufacturing_tolerances, panel_degradation_params, simulation_year)
	table.placePanels("l","w")
	
	mppt = MPPT(inverter_params, system_design_params, table)
	mppt.assignPanels()
	
	time = [2017,6,21,14,45,0]
	dt = datetime(time[0],time[1],time[2],time[3],time[4],time[5])
	
	mppt.generateIVCurve(simulation_year, dt)
	mppt_data = mppt.getMPP()
	
	mppt.plotPowerCurve(mppt_data)
	
	'''

	def __init__(self, inverter_params, design_params, table):
		self.inverter_params = inverter_params
		self.design_params = design_params
		self.table = table
		self.table_pool = []
		self.table_pool.append(table)
		self.mppt_size = float(self.inverter_params[self.AC_SIZE]) / float(self.inverter_params[self.NUM_MPPTS])
		self.overbuild_ratio = float(self.design_params[self.OVERBUILD_RATIO])
		self.dc_size = self.mppt_size * self.overbuild_ratio
		self.createTablePool()
		self.strings = []
		self.currents = []
		self.voltages = []
		self.totalIndividualPower = 0.0
		self.mppt_not_full = True
		self.mpp_tracker = MPPTTracker(inverter_params)

	def createTablePool(self):
		if self.TABLE_POOL_SIZE > 1:
			for table_count in range(self.TABLE_POOL_SIZE - 1):
				self.table_pool.append(copy.deepcopy(self.table))

	def assignPanels(self):
		if not self.design_params[self.HAS_ASSIGNED_STRINGS]:
			string_size = float(self.table_pool[0].params[self.SUBMODULES]) * float(self.table_pool[0].params[self.RATED] * self.table.string_length)
			num_strings_per_mppt = int(self.dc_size * 1000.0 / string_size)
			num_strings_per_table = self.table.num_strings
			num_tables_per_mppt = int(num_strings_per_mppt / num_strings_per_table) + 1
			for table in self.table_pool:
				table.set_loop()
			for string_count in range(num_strings_per_mppt):
				table_index = int(string_count / num_strings_per_table) % self.TABLE_POOL_SIZE
				next_table = self.table_pool[table_index]
				next_string = next_table.loop()
				for module in next_string:
					module.resetEnvironment()
				self.strings.append(next_string)

	def resetEnvironment(self):
		for table in self.table_pool:
			table.resetEnvironment()

	def generateIVCurve(self, year, dt):
		self.totalIndividualPower = 0.0
		self.currents = []
		unset_voltages = True
		for table in self.table_pool:
			table.getPOAIrradiance(dt)
		for string in self.strings:
			if unset_voltages:
				self.voltages = np.array(string.stringVoltages)
				unset_voltages = False
			string.resetEnvironment()
			string.getIVCurve(year)
			self.currents.append(string.stringCurrents)
			self.totalIndividualPower += string.totalIndividualPower
		self.allCurrents = np.array(self.currents)
		self.totalCurrent = np.sum(self.allCurrents, axis=0)

	def getMPP(self):
		mpp = self.mpp_tracker.getMPP(self.voltages,self.totalCurrent)
		return mpp

	def plotPowerCurve(self, mpp_data, time=None, irrdata=None):
		try:
			maxp = np.max(np.multiply(self.totalCurrent, self.voltages))
			non_zero_voltages = self.voltages[np.argwhere(self.totalCurrent > 0)]
			if non_zero_voltages.size == 0:
				return
			maxv = max(np.max(non_zero_voltages), self.inverter_params[self.VMPPMAX])
			mismatch = 1.0 - maxp / self.totalIndividualPower
			vjump = 20.0
			if maxv > 200:
				vjump = 50.0
			if maxv > 500:
				vjump = 100.0
			pjump = 20.0
			if maxp > 500:
				pjump = 100.0
			if maxp > 1000:
				pjump = 200.0
			if maxp > 4000:
				pjump = 1000.0
			plt.title('MPPT Power vs Voltage')
			plt.plot(self.voltages, self.voltages*self.totalCurrent)
			plt.axhline(y=self.inverter_params[self.AC_SIZE] / self.inverter_params[self.NUM_MPPTS] * 1000.0, color='r', linestyle='-')
			plt.axvline(x=self.inverter_params[self.VMPPMIN], color='r', linestyle='-')
			plt.axvline(x=self.inverter_params[self.VMPPMAX], color='r', linestyle='-')
			plt.ylim(0, 600)
			v_max = vjump * (round(maxv / vjump) + 1.0)
			p_max = pjump * (round(maxp / pjump) + 2.0)
			# Remove this later. Just for the video to keep the y-axis steady
			p_max = 6000
			plt.axis([0.0, v_max, 0.0, p_max])
			plt.text(maxv / 20.0, p_max * 0.95, 'Time: {:02d}:{:02d}'.format(time.hour, time.minute))
			plt.text(maxv / 20.0, p_max * 0.90, 'Mismatch: {0:.2f} %'.format(mismatch * 100.0))
			plt.text(maxv / 20.0, p_max * 0.85, 'MPPV: {:6.2f} V'.format(mpp_data[0]))
			plt.text(maxv / 20.0, p_max * 0.80, 'MPPP: {:6.2f} kW'.format(mpp_data[2]))
			#plt.show()
			print('Results/' + str(time.year) + '_' + str(time.month) + '_' + str(time.day) + '_' + str(time.hour) + '_' + str(time.minute).zfill(2) + '.png')
			plt.savefig('Results/' + str(time.year) + '_' + str(time.month) + '_' + str(time.day) + '_' + str(time.hour) + '_' + str(time.minute).zfill(2) + '.png')
			plt.close()
		except Error as e:
			print(e)

	def plotIVCurve(self):
		maxi = np.max(self.totalCurrent)
		maxv = 700.0
		vjump = 20.0
		if maxv > 200:
			vjump = 50.0
		if maxv > 500:
			vjump = 100.0
		ijump = 0.5
		if maxi > 2:
			ijump = 1.0
		if maxi > 10:
			ijump = 2.0
		if maxi > 20:
			ijump = 4.0
		plt.plot(self.voltages, self.totalCurrent)
		plt.axis([0.0, vjump * (round(maxv / vjump) + 1.0), 0.0, ijump * (round(maxi / ijump) + 2.0)])
		plt.show()

class MPPTTracker(Model):

	'''

	This is the core, base class for MPPTTracker algorithms. The basic functionality is to ensure that the
	voltage falls within the minimum and maximum MPPT voltage range, and that the power falls within the
	minimum and maximum power range.

	If the power exceeds the maximum power, Pmax, the we look for all voltages where the power just crosses
	the maximum power line. If there are no voltages that fall within the min and max MPP voltages, VMPPmin and VMPPmax
	the drop the power to 0 - we are clipping here.

	If there are voltages that cross the max power line, then choose the left-most of these, since this is where
	the voltage/power curve is the most stable - has the shallowest slope

	This base class is to be extended with more complex functionality in extending classes.
	Only one method should be present, a method taking a pair of numpy arrays holding voltages and currents
	and return the voltage and current corresponding to the correct MPPT point.

	'''

	def __init__(self, inverter_params):
		self.VMPPmin = inverter_params[self.VMPPMIN]
		self.VMPPmax = inverter_params[self.VMPPMAX]
		self.Pmin = inverter_params[self.PMIN] * 1000.0 / inverter_params[self.NUM_MPPTS]
		self.Pmax = inverter_params[self.AC_SIZE] * 1000.0 / inverter_params[self.NUM_MPPTS]  # This will need to be changed, cant set MPPT size by AC size

	def getMPP(self, voltages, currents):
		powers = np.multiply(currents, voltages)
		# The next three lines give the allowable values which are within the voltage limits (VMPPmin, VMPPmax)
		voltage_allowable_voltages = voltages[(np.where((voltages >= self.VMPPmin) & (voltages <= self.VMPPmax))[0])]
		voltage_allowable_powers = powers[(np.where((voltages >= self.VMPPmin) & (voltages <= self.VMPPmax))[0])]
		voltage_allowable_currents = currents[(np.where((voltages >= self.VMPPmin) & (voltages <= self.VMPPmax))[0])]
		# See if any of these voltage allowed points are below the maximum power and retrieve all allowed power values
		allowable_powers = voltage_allowable_powers[(np.where(voltage_allowable_powers < self.Pmax))]
		if allowable_powers.any():
			# If there are any of these values, then get the maximum value of these
			maxp = np.max(allowable_powers)
			# Then, find all of the voltages and currents that correspond to this allowed power range
			allowable_voltages = voltage_allowable_voltages[(np.where(voltage_allowable_powers < self.Pmax))]
			allowable_currents = voltage_allowable_currents[(np.where(voltage_allowable_powers < self.Pmax))]
			# And then find the voltage and current which give this maximum power value in the range
			index = np.where(powers == maxp)[0][0]
			voltage = voltages[index]
			current = currents[index]
			# Now, check if we have crossed the max power line
			# If so, and we are not at the edge of the MPP voltage range, then interpolate our voltages and currents
			# to equal the values they would have at the crossing
			if index < len(powers) - 1 and powers[index+1] > self.Pmax and voltages[index+1] <= self.VMPPmax:
				factor = (self.Pmax - powers[index])/(powers[index+1] - powers[index])
				voltage = voltages[index] + factor * (voltages[index+1] - voltages[index])
				current = currents[index] + factor * (currents[index + 1] - currents[index])
				maxp = self.Pmax
			elif index > 0 and powers[index-1] > self.Pmax and voltages[index-1] >= self.VMPPmin:
				factor = (self.Pmax - powers[index])/(powers[index-1] - powers[index])
				voltage = voltages[index] + factor * (voltages[index-1] - voltages[index])
				current = currents[index] + factor * (currents[index-1] - currents[index])
				maxp = self.Pmax
			return (voltage, current, maxp)
		else:
			# Otherwise, if there are no such allowed values, then we are definitely clipping
			return (self.VMPPmin, 0.0, 0.0)
 
