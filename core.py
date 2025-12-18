from scipy.interpolate import interp1d
from scipy import special
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from abc import ABC, abstractmethod
import os

class Model(object):

    '''

    This class is just a thin wrapper over the base 'object' Python class
    where application level assumptions will be stored
    These of course are called from the application classes which all extend Model
    For example, simply by calling self.vp from any class we can find the number of points along the voltage axis.

    '''

    vp  = 51                            # panel voltage points - the number of points along the x or voltage axis to model (add 1 to the number of intervals needed, so 201 for 200 intervals)
    ip  = 2001                          # panel current points - the number of points along the y or current axis to model (same as above, add 1)
    mi  = 20.0                          # the maximum panel current to model
    ipr = mi/float(ip-1)                # the precision of the points along the current axis - how far apart they are
    TABLE_POOL_SIZE = 3                 # the number of tables to create in an MPPT table pool object

    maxSimTime = 20                     # The maximum simulation time in years
    rt2 = 2.0**0.5                      # Root of 2 - a convenience constant
    dtr = math.pi/180.0
    
    resolution = 10  # When looking at IV curves for individual panels, set this to 10, otherwise set to 1
    
    maxStringVoltage    = 2000          # The maximum possible string voltage
    stringVoltagePoints = 1000 * resolution + 1          # The number of interpolation points for the maxStringVoltage - gives us a voltage discrete size of maxStringVoltage/(stringVoltagePoints-1)  - we add 1 because we include the final point

    I0 = 0.0         # the dark saturation current
    T0 = 298.15      # the STC temperature (K)
    Istc = 1000      # the STC irradiance (W/m2)
    bypassV = -0.6   # the panel bypass voltage

    K = 1.38e-23     # Boltzmann's constant
    q = 1.602e-19    # electron charge in Coulombs

    Eg_a = 7.021e-4  # eV/K
    Eg_b = 1108      # K
    Eg_0 = 1.1577    # eV
    K_ev = 8.617e-5  # eV/K
    I_s = 1.5e5      # A/cm2
    cell_area = 15.6*15.6*72 # cm2


    NCELLS = 0  # position constants for parameters list
    VOC = 1
    ISC = 2
    RS = 3
    RSH = 4
    H = 5
    W = 6
    NAME = 7
    RATED = 8
    SUBMODULES = 9
    KI = 10
    KV = 11
    N = 12

    INVERTER_NAME = 0
    AC_SIZE = 1
    NUM_MPPTS = 2
    PMIN = 3
    VMPPMIN = 4
    VMPPMAX = 5

    OVERBUILD_RATIO = 0
    HAS_ASSIGNED_STRINGS = 1
    ASSIGNED_STRINGS = 2                  # A dictionary giving the numbers of strings assigned to each MPPT: \
    #  {inverter_1: [num_mppt1, num_mppt2...], inverter_2: [num_mppt1, num_mppt2...], ...}

    PARAMETER_INPUT_MODE = 0
    PARAMETER_SIMULATE_MODE = 1

    GLASS_CELL_POLYMER_OPEN = 0
    
    RADBINS = '/home/geoffrey/Backup/Install/radiance-5.2.a46558bb5f-Linux/usr/local/radiance/bin/'
    
    ONE_PORTRAIT  = 1
    TWO_LANDSCAPE = 2

class ThermalModel(ABC,Model):

    '''
    This is the abstract base class (ABC) for all thermal models. All must have a getT method.
    '''

    @abstractmethod
    def getT(self,*args):
        pass

class SandiaModel(ThermalModel):

    E0 = 1000.0

    def __init__(self, mountingType):
        self.mountingType = mountingType
        if self.mountingType == self.GLASS_CELL_POLYMER_OPEN:
            self.a = -3.56
            self.b = -0.0750
            self.dT = 3

    def getT(self,*args):
        I = args[0]
        Tamb = args[1]
        V = args[2]
        return I * math.exp(self.a + self.b * V) + Tamb + I/self.E0 * self.dT

class Environment(Model):

    Iplane = 1000.0						# the STC irradiation (W/m2)
    T = 298.15							# the STC temperature (K)
    V = 1.0                             # the wind speed at 10m height (m/s)
    soiling = 0.0

    def __init__(self,*args):
        if len(args) == 2:
            self.Iplane = args[0]
            self.T = args[1]
            self.soiling = 0.00
        if len(args) == 3:
            self.Iplane = args[0]
            self.T = args[1]
            self.soiling = args[2]
        if len(args) == 4:
            self.Iplane = args[0]
            self.T = args[1]
            self.soiling = args[2]
            self.V = args[3]


    def __str__(self):
        return "POA: {:7.2f} W/m2, T: {:6.2f} K, Soiling: {:4.2f}%, V: {:4.2f}m/s".format(self.Iplane, self.T, self.soiling*100.0, self.W)

    def setIplane(self,Iplane):
        self.Iplane = Iplane

    def setT(self,T):
        self.T = T

    def setSoiling(self,soiling):
        self.soiling = soiling

    def setV(self,V):
        self.V = V

class EnvironmentGenerator(Model):

    '''

    This class is used to model varying environmental conditions that are relevant to panel operation - irradiance and temperature.
    The class is initialized by passing it a temperature mean and standard deviation, and an irradiance mean ans standard deviation.
    Calling getEnvironment returns an instance of an Environment class, which is used by Panels to hold panel operating conditions.

    Todo: Right now, only an unvarying wind speed is allowed. Fix this later.

    '''

    Tm = 0
    Ts = 0
    Im = 0
    Is = 0
    rt2 = 2.0**0.5

    def __init__(self,Tm,Ts,Im,Is,Imax,Imin,soiling,V):
        self.Tm = Tm                   # mean temperature
        self.Ts = Ts                   # standard deviation of temperature
        self.Im = Im                   # mean irradiance
        self.Is = Is                   # standard deviation of irradiance
        self.Imin = Imin               # The minimum possible irradiance - eg. what you might get from the thickest part of a cloud layer
        self.Imax = Imax               # The maximum possible irradiance - eg. lensing effects at cloud edges
        self.soiling = soiling         # degree of soiling from 0.0 not to 1.0 total soiling
        self.V = V                     # wind speed at 10m height
        self.thermalModel = SandiaModel(self.GLASS_CELL_POLYMER_OPEN) # this will be the default thermal model, but reset if needed using setThermalModel

    def setGenerationMode(self, irradianceMode, thermalMode, windMode):
        self.irradianceMode = irradianceMode
        self.thermalMode = thermalMode
        self.windMode = windMode

    def setThermalModel(self,thermalModel):
        self.thermalModel = thermalModel
        
    def setIm(self, Im):
        self.Im = Im
        
    def setIs(self, Is):
        self.Is = Is

    def getEnvironment(self):
        ir = random.random()
        # We have to make sure that the irradiation does not drop below the minimum irradiation
        # Without ensuring against this, it is possible to have a negative irradiance
        # Also, make sure that irradiance doesn't go above the maximum
        I = min(self.Imax, max(self.Im + self.rt2*self.Is*special.erfinv(2.0 * ir - 1.0), self.Imin))
        tr = random.random()
        if self.thermalMode == self.PARAMETER_INPUT_MODE:
            T = self.Tm + self.rt2*self.Ts*special.erfinv(2.0 * tr - 1.0)
        else:
            Tamb = self.Tm + self.rt2*self.Ts*special.erfinv(2.0 * tr - 1.0)
            T = self.thermalModel.getT(I, Tamb, self.V)
        outEnv = Environment(I, T, self.soiling)
        outEnv.setSoiling(self.soiling)
        return outEnv

class DegradationModel(Model):

    '''
    This class is the panel degradation model. Degradation is effected through changes to the open circuit voltage (Voc) and short circuit
    current (Isc) as well as the series (Rs) and shunt (Rsh) resistances.
    Series resistances will increase representing a harder path for current to flow through to the load, and shunt resistances
    will decrease, representing an easier path for the current to leak through the circuit, bypassing the load entirely.
    Values for Voc, Isc, Rs and Rsh will be generated randomly over an N year period.

    The baseParams are the base parameters that describe the module.
    This class will generate a series of N series and shunt resistances and pass this back to the Panel object, so that the panel
    object can know how its performance will decrease over time when the panel obejct is created.

    This model will use an exponential degradation model:

    XFac = Parameter X (X one of voc, isc, ser or shn) Increase Factor: How much per year X increases - multiply last years value by this factor, so this is an exponential model
    XFac = Normal(X_mean,X_std)


    The degradationParameters passed to the constructor are [ser_mean,ser_std,shn_mean,shn_std]

    '''

    def __init__(self,degradationParams,N):
        self.voc_mean = float(degradationParams[0])
        self.voc_std  = float(degradationParams[1])
        self.isc_mean = float(degradationParams[2])
        self.isc_std  = float(degradationParams[3])
        self.ser_mean = float(degradationParams[4])
        self.ser_std = float(degradationParams[5])
        self.shn_mean = float(degradationParams[6])
        self.shn_std = float(degradationParams[7])
        self.degradationParams = degradationParams
        self.N = N
        self.factors = np.zeros((self.N+1,4))                       # We will have N+1 values here, the year 0 value and the values for the first N years

    def getDegradationFactors(self):
        serFac = 1.0
        shnFac = 1.0
        vocFac = 1.0
        iscFac = 1.0
        self.factors[0,:] = [vocFac,iscFac,serFac,shnFac]
        for n in range(self.N):
            random.seed(time.time())
            ser_rand = random.random()
            shn_rand = random.random()
            voc_rand = random.random()
            isc_rand = random.random()
            self.factors[n+1,0] = self.factors[n,0]*(self.voc_mean + self.rt2*self.voc_std*special.erfinv(2.0 * voc_rand - 1.0))
            self.factors[n+1,1] = self.factors[n,1]*(self.isc_mean + self.rt2*self.isc_std*special.erfinv(2.0 * isc_rand - 1.0))
            self.factors[n+1,2] = self.factors[n,2]*(self.ser_mean + self.rt2*self.ser_std*special.erfinv(2.0 * ser_rand - 1.0))
            self.factors[n+1,3] = self.factors[n,3]*(self.shn_mean + self.rt2*self.shn_std*special.erfinv(2.0 * shn_rand - 1.0))
        return self.factors

class Panel(Model):

    '''

    This class creates instances of Panels, used to model PV panels.
    Panel and environmental characteristics are set by passing a list of panel parameters and an Environment instance to the constructor.

    Usage:

    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    degradationParams = [0.998,0.0002,0.993,0.0002,1.01,0.1,0.975,0.015]
    env = Environment(1000,298.15)
    panel = Panel(params,env)                                        # Set the module params and the environment
    panel.setDegradationModel(degradationParams,20)                  # Set the degradation and pre-generate performance degradation over 20 years
    panel.generateIVCurve(20)                                        # Generate the IV curve for this module at year 20
    panel.plotIVCurve()                                              # Plot the curve

    '''

    def __init__(self, params, eg, env=None):

        self.params = params
        self.setParams(params)
        self.eg = eg
        if not env:
            env = eg.getEnvironment()
        else:
            self.env = env

        self.ivcurve = np.zeros((self.vp,2))                            # Create an empty numpy array to hold a the IV curve of the panel where the voltages have constant jumps of size Model.vp - V is the independent variable in the 5 variable 1-diode equation so we need constant jumps in V
        self.currents = np.linspace(0.0,self.mi,self.ip)                # For adding panels in series though, we need constant jumps in current (so that we can add voltages of the same current in different panels), so we need this array to hold the constant jump currents
        self.voltages = np.zeros((self.ip))                             # This array will hold the voltages that result from interpolating the voltages from self.ivcurve using the values from the self.currents array mentioned in the line above - doing this, we can add voltages in these arrays element by element since the corresponding currents are all the same

        self.setEnv(env)
        self.degradation = None
        self.degradationModelSet = False
        self.year = 0
        self.shading_factor = 1.0
        
        self.bypass = self.bypassV
    
    def initialize(self):
        self.ivcurve = np.zeros((self.vp,2))                            # Create an empty numpy array to hold a the IV curve of the panel where the voltages have constant jumps of size Model.vp - V is the independent variable in the 5 variable 1-diode equation so we need constant jumps in V
        self.voltages = np.zeros((self.ip))                             # This array will hold the voltages that result from interpolating the voltages from self.ivcurve using the values from the self.currents array mentioned in the line above - doing this, we can add voltages in these arrays element by element since the corresponding currents are all the same

    def __str__(self):
        (vmpp, impp, pmpp) = self.getMppParams()
        return "Year: {:=2d}, Env: {}, Shading: {:5.3f}, Vmpp: {:7.4f} V, Impp: {:7.4f} A, Pmpp: {:8.4f} W".format(self.year, self.env, self.shading_factor, vmpp, impp, pmpp)

    def __iter__(self):
        return self

    def setParams(self, params):
        self.Ncells = params[self.NCELLS]
        self.Voc = params[self.VOC]
        self.Isc = params[self.ISC]
        self.Rs = params[self.RS]
        self.Rsh = params[self.RSH]
        self.w = params[self.W]
        self.h = params[self.H]
        self.name = params[self.NAME]
        self.rated = params[self.RATED]
        self.submodules = params[self.SUBMODULES]
        self.Kv = params[self.KV]
        self.Ki = params[self.KI]
        self.n = params[self.N]

    def setDegradationModel(self, degradationParams, N):
        self.N = N
        self.degradationParams = degradationParams
        dm = DegradationModel(degradationParams,N)
        self.degradation = dm.getDegradationFactors()
        self.degradationModelSet = True

    def setEnv(self, environment):
        self.env = environment
        self.I0 = self.Izero(environment.T)

    def resetEnvironment(self):
        env = self.eg.getEnvironment()
        self.setEnv(env)

    def setShadingFactor(self, shading_factor):
        self.shading_factor = shading_factor
        
    def initializeShadingFactor(self):
        self.setShadingFactor(1.0)
        
    def getShadingFactor(self):
        return self.shading_factor

    def getNumberOfPanelsPerModule(self):
        return self.submodules

    def Izero(self, T):
        Isct = self.Isc + self.Ki * (T - self.T0)
        Voct = self.Voc + self.Kv * (T - self.T0)
        I0 = Isct / (math.exp((self.q * Voct) / (self.Ncells * self.n * self.K * T)) - 1.0)
        return I0

    def getI0(self):
        T = self.env.T
        a = self.Eg_a
        b = self.Eg_b
        Eg_0 = self.Eg_0
        K = self.K_ev
        Eg = Eg_0 - (a*T**2.0)/(b + T)
        return self.I_s*math.exp(-1.0*Eg/(K*T))*self.cell_area

    def secant(self, x0, x1, V, n):
        myf = self.f
        for i in range(n):
            if myf(x1,V)-myf(x0,V) < 0.01:
                return x1
            x_temp = x1 - (myf(x1,V)*(x1-x0)*1.0)/(myf(x1,V)-myf(x0,V))
            x0 = x1
            x1 = x_temp
        return x1

    def generateIVCurve(self, year):
        # FIX THIS IF THERE ARE PROBLEMS........................................................
        #self.initialize()
        mysecant = self.secant
        self.year = year
        if not self.degradationModelSet:
            self.degradation = np.ones((self.maxSimTime,4))                         # No degradation model has been set, so just set maxSimTime years of no change in series or shunt resistances
            self.degradationModelSet = True
        self.Voc = self.params[self.VOC] * self.degradation[year,0]
        self.Isc = self.params[self.ISC] * self.degradation[year,1]
        self.Rs = self.params[self.RS] * self.degradation[year,2]
        self.Rsh = self.params[self.RSH] * self.degradation[year,3]
        for i in range(self.vp):
            V = self.Voc*(i+1.0)/float(self.vp)
            I = mysecant(0,self.Isc*self.env.Iplane*self.shading_factor*(1.0-self.env.soiling)/self.Istc,V,100)
            self.ivcurve[i,0] = V
            self.ivcurve[i,1] = I
        maxi = np.max(self.ivcurve[:,1])
        indxi = int(maxi/self.ipr)
        if self.ivcurve[-1,1] > 0.0:
            # Sometimes the secant method gives us a current greater than 0 which throws off our later fine-grained interpolation near 0.0 A - the interpolator cannot extrapolate and fails - so extend the curve linearly down to 0.0 A for positive final currents to avoid this
            slope = (self.ivcurve[-1,0]-self.ivcurve[-2,0])/(self.ivcurve[-1,1]-self.ivcurve[-2,1])
            delta_v = -1.0*self.ivcurve[-1,1]*slope
            self.ivcurve[-1,1] = 0.0
            self.ivcurve[-1,0] = self.ivcurve[-1,0] + delta_v
        x = self.ivcurve[:,1]
        y = self.ivcurve[:,0]
        revx = x[::-1]
        revy = y[::-1]
        interpolator = interp1d(revx,revy,kind='linear')  # Need to reverse x and y since x has to be monotonic for the interpolator to work
        new_currents = self.currents[0:indxi]
        self.voltages[0:indxi] = interpolator(new_currents)
        self.voltages[indxi:] = self.bypass

    def f(self,x,V):
        I0 = self.getI0()
        return x - self.Isc*self.env.Iplane*self.shading_factor/self.Istc + I0*(math.exp((self.q*(V+x*self.Rs))/(self.Ncells*self.n*self.K*self.env.T)) - 1.0) + (V + x*self.Rs)/self.Rsh

    def getMppParams(self):
        ivcurve = self.ivcurve
        (maxv,maxi) = max(ivcurve, key=lambda x:x[0]*x[1])
        return (maxv,maxi,maxv*maxi)

    def plotIVCurve(self):
        plt.plot(self.ivcurve[:,0],self.ivcurve[:,1])
        (v,i,p) = self.getMppParams()
        plt.scatter(v,i)
        plt.title("IV Curve for " + self.name + " at " + str(self.env.Iplane) + " W/m2 and " + str(self.env.T) + " K")
        plt.xlabel("Panel voltage (V)")
        plt.ylabel("Panel current (A)")
        plt.axis([0.0,5.0*(round(self.Voc/5.0)+1.0),0.0,(round(i)+1.0)])
        plt.text(self.Voc/2.0, i-1.0,"MPP: (" + '{0:.2f}'.format(v) + "V, " + '{0:.2f}'.format(i) + "A, "+ '{0:.2f}'.format(p) + " W)")
        plt.show()

    def plotPowerCurve(self):
        (v,i,p) = self.getMppParams()
        plt.scatter(self.submodules*v,self.submodules*p)
        plt.title("Power Curve for Panel at " + str(self.env.Iplane) + " W/m2 and " + str(self.env.T) + " K")
        plt.xlabel("Panel voltage (V)")
        plt.ylabel("Panel power (W)")
        plt.axis([0.0,5.0*(round(self.submodules*self.Voc/5.0)+1.0),0.0,10.0*(round(self.submodules*p/10.0)+1.0)])
        plt.plot(self.submodules*self.ivcurve[:,0],self.submodules*self.ivcurve[:,0]*self.ivcurve[:,1])
        plt.text(self.submodules*self.Voc/10.0, self.submodules*p*0.9,"MPP: (" + '{0:.2f}'.format(self.submodules*v) + "V, " + '{0:.2f}'.format(self.submodules*p) + "W)")
        plt.show()

class Cell(Model):
    
    '''
    
    This class represents a single cell in a module, with no bypass diode functionality.
    The parameters to pass to the class are the same parameters used for the Panel class.
    This class will make the necessary conversions.
    
    Usgage:
    
    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 40.0, 1100.0, 800.0, 0.02, 3.0)
    eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
    cell = Cell(params, eg, None)
    cell.generateIVCurve()
    currents, voltages = cell.getCurrentsVoltages()
    plt.plot(voltages, currents*voltages)
    plt.show()
    
    '''
    
    def __init__(self, params, eg, env):
        cellParams = []
        ncells = params[0]
        cellParams.append(1)
        cellParams.append(params[1]/ncells)
        cellParams.append(params[2])
        cellParams.append(params[3]/ncells)
        cellParams.append(params[4]*ncells)
        cellParams.append(1)
        cellParams.append(1)
        cellParams.append('Cell')
        cellParams.append(params[8]/ncells)
        cellParams.append(1)
        cellParams.append(params[10])
        cellParams.append(params[11])
        cellParams.append(params[12])
        self.cell = Panel(cellParams, eg, env)
        self.cell.bypass = 0.0  # For individual cells, there are no bypass diodes so set this to 0
        
    def setIPlane(self, iplane):
        self.cell.env.setIplane(iplane)
        
    def generateIVCurve(self):
        self.cell.generateIVCurve(0)
        
    def getCurrentsVoltages(self):
        return (self.cell.currents, self.cell.voltages)        
        
class PanelFactory(Model):

    def __init__(self, eg, params, manufacturingParams, tolerances):
        self.eg = eg
        self.params = params
        self.manufacturingParams = manufacturingParams
        self.tolerances = tolerances
        self.stc = Environment()

    def getPanel(self):
        outOfTolerance = True
        panel = None
        while outOfTolerance:
            newParams = copy.copy(self.params)
            voc_mean = self.manufacturingParams[0]
            voc_std = self.manufacturingParams[1]
            isc_mean = self.manufacturingParams[2]
            isc_std = self.manufacturingParams[3]
            random.seed(time.time())
            voc_rand = random.random()
            isc_rand = random.random()
            newParams[4] = newParams[4] * (voc_mean + self.rt2 * voc_std * special.erfinv(2.0 * voc_rand - 1.0))
            newParams[5] = newParams[5] * (isc_mean + self.rt2 * isc_std * special.erfinv(2.0 * isc_rand - 1.0))
            panel = Panel(newParams, self.eg, self.stc)   # This is a new Panel, so the environment has to be STC conditions - we are checking to see if it is withing manufacturing tolerances
            panel.generateIVCurve(0)
            (v, i, p) = panel.getMppParams()
            if (p > (1.0 - self.tolerances[0] / 100.0) * panel.rated) and (p < (1.0 + self.tolerances[1] / 100.0) * panel.rated):  # OK, this panel is within manufacturing tolerance, set the proper environment (not STC conditions) and mark it within manufacturing tolerance
                env = self.eg.getEnvironment()
                panel.setEnv(env)
                outOfTolerance = False
        return panel

class PanelString(Model):
    '''

    This class models a string of panels. To initialize this class, a list of panel parameters, and Environment Generator and the number of panels in the string must be passed to the constructor.
    Note that the EnvironmentGenerator must represent how the environmental conditions will vary among the panels of the string, so variations must be representative of panel level variations along a single string.
    Usage is as follows.

    Note that if one wants to simulate a new varying panel environment (new randomly chosen irradiation and temperature conditions) on the same
    panel string, with the same panel properties, then it is only necessary to call:

    PanelString.resetEnvironment()
    PanelString.getIVCurve(year)

    With these calls, only the environment will have changed. Note of course that the year does not have to be the same, so degradation
    can be from another year.

    Usage:

    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    degradationParams = [0.998, 0.0002, 0.993, 0.0002, 1.01, 0.1, 0.975, 0.015]
    manufacturingParams = [1.0, 0.0072, 1.0, 0.0134]  # These ratios are rough estimates from the paper - Reis et al, 2002
    tolerances = [0.0, 5.0 / 315.0 * 100.0]  # Typical manufacturing tolerance 315 W (-0W/+5W)
    eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 40.0, 1100.0, 800.0, 0.02, 3.0)
    eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
    PS = PanelString(eg, params, manufacturingParams, tolerances, 54)
    PS.setDegradationModel(degradationParams, 20)  # Generate a DegradationModel to generate and store all of the degradation charactersistics of the panels up to year 20
    PS.createPanels()  # Now use a PanelFactory to create all of the panels and assign them to this String
    PS.getIVCurve(0)  # Generate an IV curve for this string at year 10
    PS.plotPowerCurve()
    PS.resetEnvironment()  # Generate a new environment - irradiance and temperature - say for the next time step
    PS.getIVCurve(10)  # Get a new IV curve for the new environment at year 10
    PS.plotPowerCurve()

    '''

    def __init__(self, eg, params, manufacturingParams, tolerances, size):
        self.size = size
        self.eg = eg
        self.params = params
        self.manufacturingParams = manufacturingParams
        self.tolerances = tolerances
        self.ivcurves = np.zeros((size, self.ip))
        self.ivcurve = np.zeros((self.ip, 2))
        self.string = []
        self.currents = []  # MPP max power point currents individually for panels in the string
        self.voltages = []  # MPP max power point voltages individually for panels in the string
        self.powers = []    # MPP max power point powers individually for panels in the string
        self.totalIndividualPower = 0
        self.mincur = 0.0
        self.stringVoltages = np.linspace(0.0, self.maxStringVoltage, self.stringVoltagePoints, dtype=float,
                                          endpoint=True)  # We want evenly spaced voltages for the string currents
        self.stringCurrents = np.zeros((self.stringVoltagePoints))

    def initialize(self):
        self.ivcurves = np.zeros((self.size, self.ip))
        self.ivcurve = np.zeros((self.ip, 2))
        self.stringCurrents = np.zeros((self.stringVoltagePoints))
        
    def __iter__(self):
        self.iteration_index = 0
        return self

    def __str__(self):
        output = "[\n"
        output += "    String size: {0} panels =>\n".format(len(self.string))
        for index, panel in enumerate(self.string):
            output += "    [ Panel[{:=2d}]: {} ]\n".format(index,str(panel))
        output = output[:-5] + "\n]"
        return output

    def __next__(self):
        if self.iteration_index == len(self.string):
            raise StopIteration
        else:
            lower  = self.string[self.iteration_index]
            middle = self.string[self.iteration_index + 1]
            upper  = self.string[self.iteration_index + 2]
            self.iteration_index = self.iteration_index + 3
            return Module(lower, middle, upper)

    def setDegradationModel(self, degradationParams, N):
        self.degradationParams = degradationParams
        self.N = N

    def createPanels(self):
        pf = PanelFactory(self.eg, self.params, self.manufacturingParams, self.tolerances)
        self.totalIndividualPower = 0.0
        for i in range(self.size):
            #env = self.eg.getEnvironment() ###
            #myPanel = pf.getPanel()        ###
            #nextPanel = copy.deepcopy(myPanel) ###  # Simply using nextPanel = pf.getPanel(env) creates a list of identical copies of panel - this is the only way to get around this. Try to figure this out - thre is some sort of scoping issue which I don't understand
            nextPanel = pf.getPanel()
            nextPanel.setDegradationModel(self.degradationParams, self.N)
            self.string.append(nextPanel)

    def resetEnvironment(self):
        self.stringCurrents = np.zeros((self.stringVoltagePoints))  # We are resetting the environment, so we want to clean out the old IV curve first
        for panel in self.string:
            env = self.eg.getEnvironment()
            panel.setEnv(env)

    def getString(self):
        return self.string

    def getIVCurve(self, year):
        count = 0
        panelCurrents = None
        self.totalIndividualPower = 0.0
        self.currents = []  # MPP max power point currents individually for panels in the string
        self.voltages = []  # MPP max power point voltages individually for panels in the string
        self.powers = []    # MPP max power point powers individually for panels in the string
        for panel in self.string:
            panel.generateIVCurve(year)
            panelCurrents = panel.currents
            (vmpp, impp, pmpp) = panel.getMppParams()
            self.totalIndividualPower += pmpp
            self.currents.append(impp)
            self.voltages.append(vmpp)
            self.powers.append(pmpp)
            self.ivcurves[count, :] = panel.voltages
            count += 1
        # Now, sum up all the voltages for all the panels in the string and get the currents which are the same for each panel
        self.ivcurve[:, 0] = np.sum(self.ivcurves, axis=0)
        self.ivcurve[:, 1] = panelCurrents
        # Rename the currents and total string voltage for interpolation
        x = self.ivcurve[:, 0]
        y = self.ivcurve[:, 1]
        # Reverse the voltages and currents so that the currents are monotonic, this is required by the interpolation algorithm
        revx = x[::-1]
        revy = y[::-1]
        # Find out where the voltage crosses the y axis -> in other words becomes zero
        # There are a lot of voltage values at the reversed bias voltage -> diode reverse bias times number of panels in string
        # We want to ignore these values when we do our interpolation, so find the index where the voltage becomes zero
        # If the string is heavily shaded however, such a crossing won't occur, so if so, just send back the np.zeros version of the currents (try/except)
        try:
            sign_change = np.where(np.diff(np.sign(revx)))[0][0]  # How to find the index when a numpy array changes sign
            revx = revx[sign_change:]
            revy = revy[sign_change:]
            # At the sign change index, revx was still equal to the reverse bias value, so set the voltage to zero where this crossing occurs
            revx[0] = 0
            # Find out what the maximum voltage is so that we don't exceed this value with our regularly spaced voltage values that we want to interpolate to
            # These regularly spaced values are self.stringVoltages
            maxx = max(revx)
            # Now, get the interpolator from the evenly spaced currents and dependent voltages
            interpolator = interp1d(revx, revy, kind='linear')  # Need to reverse x and y since x has to be monotonic for the interpolator to work
            interpVoltages = self.stringVoltages[np.where(self.stringVoltages < maxx)]  # How to get evenly spaced voltages ensuring we don't exceed maxx - which would cause an interpolation error
            numPoints = len(interpVoltages)  # Find out how many points are in our evenly spaced voltages less than our maximum value in our curve to interpolate
            stringCurrents = interpolator(interpVoltages)
            self.stringCurrents[0:numPoints] = stringCurrents
        except IndexError as e:
            print(e)
            print("Problem in PanelString.getIVCurve() !!!!!!!")

    def plotIVCurve(self):
        maxi = np.max(self.ivcurve[:, 1])
        maxv = np.max(self.ivcurve[:, 0])
        vjump = 20.0
        if maxv > 200:
            vjump = 50.0
        positive = [1 if x > 0 else 0 for x in
                    self.ivcurve[:, 0]]  # gives 1 if the voltage is positive, and 0 if voltage is negative
        mcfpv = max([p * i for (p, i) in
                     zip(positive, self.ivcurve[:, 1])])  # gives the maximum current when the voltage is positive
        plt.plot(self.ivcurve[:, 0], self.ivcurve[:, 1])
        plt.title("IV Curve for " + str(len(self.string)) + " panel string at " + str(self.eg.Im) + " +/- " + str(
            self.eg.Is) + " W/m2 and " + str(self.eg.Tm) + " +/- " + str(self.eg.Ts) + " K")
        plt.xlabel("String voltage (V)")
        plt.ylabel("String current (A)")
        plt.axis([0.0, vjump * (round(maxv / vjump) + 1.0), 0.0, 2.0 * (round(mcfpv / 2.0) + 1.0)])
        plt.show()

    def plotPowerCurve(self):
        maxv = np.max(self.ivcurve[:, 0])
        maxp = np.max([i * v for (i, v) in zip(self.ivcurve[:, 1], self.ivcurve[:, 0])])
        mismatch = 1.0 - maxp / self.totalIndividualPower
        vjump = 20.0
        if maxv > 200:
            vjump = 50.0
        pjump = 20.0
        if maxp > 500:
            pjump = 50.0
        if maxp > 1000:
            pjump = 100.0
        plt.plot(self.ivcurve[:, 0], self.ivcurve[:, 0] * self.ivcurve[:, 1])
        plt.title("Power Curve for " + str(len(self.string)) + " panel string at " + str(self.eg.Im) + " +/- " + str(
            self.eg.Is) + " W/m2 and " + str(self.eg.Tm) + " +/- " + str(self.eg.Ts) + " K")
        plt.xlabel("String voltage (V)")
        plt.ylabel("String power (W)")
        plt.axis([0.0, vjump * (round(maxv / vjump) + 1.0), 0.0, pjump * (round(maxp / pjump) + 1.0)])
        plt.text(maxv / 10.0, maxp * 0.9, "Mismatch losses: " + '{:.2f}'.format(mismatch * 100.0) + "%")
        plt.text(maxv / 10.0, maxp * 0.8, "Maximum power: " + '{:6.2f}'.format(maxp) + " W")
        plt.show()

    def writeData(self):
        with file('out.txt', 'w') as outfile:
            for i in range(self.size):
                outfile.write("Next slice\n")
                nextslice = self.ivcurves[i, :, :]
                np.savetxt(outfile, nextslice)

class StringPool(Model):

    '''

    This class creates a pool or collection of PanelStrings to assign in a grouping. Such a grouping would be used for
    situations such as when panels are assigned to a Table for calculation of shading conditions.

    The pool will create a PanelFactory object which creates panels with variations in panel properties according to manufacturing tolerances
    (see the PanelFactory class). A DegradationModel is also created to model the degradation of the panel over time, so that when getIVCurve(year)
    is called on the string, that the correct degradation will be calculated before the IV curve is calculated.

    Naturally, a panel Environment class is instantiated for each panel initially so that the fluctuating irradiance and temperature can be factored
    into the calculation of the IV curve.

    Usage:

    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    degradationParams = [0.998,0.0002,0.993,0.0002,1.01,0.1,0.975,0.015]  # These ratios are rough estimates from the paper - Reis et al, 2002
    manufacturingParams = [1.0,0.0072,1.0,0.0134]                         # These ratios are rough estimates from the paper - Reis et al, 2002
    tolerances = [0.0,5.0/315.0*100.0]	                                  # Typical manufacturing tolerance 315 W (-0W/+5W)
    eg = EnvironmentGenerator(298.15,3.0,1000.0,100.0)
    sp = StringPool(eg,params,manufacturingParams,tolerances,degradationParams,54,40)
    sp.generatePool(10)

    '''

    def __init__(self, eg, params, manufacturingParams, tolerances, degradationParams, stringSize, poolSize):
        self.eg = eg
        self.params = params
        self.manufacturingParams = manufacturingParams
        self.tolerances = tolerances
        self.degradationParams = degradationParams
        self.stringSize = stringSize
        self.poolSize = poolSize
        self.pool = []

    def __iter__(self):
        self.iteration_index = 0
        return self

    def __next__(self):
        if self.iteration_index == len(self.pool):
            raise StopIteration
        return_string = self.pool[self.iteration_index]
        self.iteration_index += 1
        return return_string

    def generatePool(self, year):
        self.pool = []
        self.year = year
        for count in range(self.poolSize):
            nextString = PanelString(self.eg, self.params, self.manufacturingParams, self.tolerances, self.stringSize)
            nextString.setDegradationModel(self.degradationParams, year)
            nextString.createPanels()
            nextString.getIVCurve(year)
            self.voltages = nextString.stringVoltages
            self.pool.append(nextString)

    def getPanels(self):
        for string in self.pool:
            for panel in string:
                return panel

class Module(Model):

    def __init__(self, lower, middle, upper):
        self.lower  = lower
        self.middle = middle
        self.upper  = upper

    def __str__(self):
        return "[ Orientation: {}, Wdith: {}, Height: {}\n Lower : {} \n Middle: {} \n Upper : {} \n]".format(self.orientation, self.x, self.y, self.lower, self.middle, self.upper)

    def __iter__(self):
        self.last_returned = None
        return self

    def __next__(self):
        if not self.last_returned:
            self.last_returned = 'lower'
            return self.lower
        elif self.last_returned == 'lower':
            self.last_returned = 'middle'
            return self.middle
        elif self.last_returned == 'middle':
            self.last_returned = 'upper'
            return self.upper
        elif self.last_returned == 'upper':
            raise StopIteration
        else:
            raise Exception('Error in Module class __next__ method! Wrong last returned reference.')

    def initializeShadingFactors(self):
        self.upper.initializeShadingFactor()
        self.middle.initializeShadingFactor()
        self.lower.initializeShadingFactor()
        
    def getNumberOfPanelsPerModule(self):
        return self.lower.getNumberOfPanelsPerModule()

    def setConfiguration(self, table, orientation, x, y):
        self.table = table
        self.orientation = orientation
        self.x = x
        self.y = y

    def resetEnvironment(self):
        self.lower.resetEnvironment()
        self.middle.resetEnvironment()
        self.upper.resetEnvironment()

class BFModule(Model):
    
    '''
    A class which models a bi-facial module.
    Pass the module parameters for the front module as for any mono-facial module.
    Using setIrradiances, pass a list of irradiances for the lower, middle, uppper sub-modules respectively for the front and back modules respectively
    Calling getIVCurve and plotIVCurve sets and plots the resulting IVCurve
        
    Usage:
        
    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    manufacturingParams = [1.0, 0.0072, 1.0, 0.0134]  # These ratios are rough estimates from the paper - Reis et al, 2002
    tolerances = [0.0, 5.0 / 315.0 * 100.0]  # Typical manufacturing tolerance 315 W (-0W/+5W)
    degradationParams = [0.998, 0.0002, 0.993, 0.0002, 1.01, 0.1, 0.975, 0.015]
    eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 0.0, 1100.0, 800.0, 0.02, 3.0)
    eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
    bf = BFModule(eg, params, manufacturingParams, tolerances, degradationParams)
    irrads = [1212.0, 1212.0, 1212.0, 532.0, 514.0, 337.0]
    bf.setIrradiances(irrads)
    bf.getIVCurve(0)
    bf.plotPowerCurve()
        
        
    '''
    
    def __init__(self, eg, params, manufacturingParams, tolerances, degradationParams, bifaciality):
        self.bifaciality = bifaciality
        self.fstring = PanelString(eg, params, manufacturingParams, tolerances, 3)
        self.fstring.setDegradationModel(degradationParams, 0)
        self.fstring.createPanels()
        for index, module in enumerate(self.fstring):
            self.front = module
        self.bstring = PanelString(eg, params, manufacturingParams, tolerances, 3)
        self.bstring.setDegradationModel(degradationParams, 0)
        self.bstring.createPanels()
        for index, module in enumerate(self.bstring):
            self.back = module
        self.reinitialize()  
        
    def reinitialize(self):
        self.voltages = None
        self.currents = None
        self.powers = None
        self.maxp = 0.0
        self.fstring.totalIndividualPower = 0.0
        self.bstring.totalIndividualPower = 0.0
        self.totalIndividualPower = 0.0
        self.fstring.initialize() 
        self.bstring.initialize()
            
    def setIrradiances(self, irradString):
        self.irradString = irradString
        self.front.lower.env.setIplane(irradString[3])
        self.front.middle.env.setIplane(irradString[4])
        self.front.upper.env.setIplane(irradString[5])
        self.back.lower.env.setIplane(irradString[0] * self.bifaciality)
        self.back.middle.env.setIplane(irradString[1] * self.bifaciality)
        self.back.upper.env.setIplane(irradString[2] * self.bifaciality)
        self.avg_back_irradiance = (irradString[3] + irradString[4] + irradString[5]) / 3.0
        self.avg_front_irradiance = (irradString[0] + irradString[1] + irradString[2]) / 3.0

    def getIVCurve(self, year):
        self.reinitialize()
        self.fstring.getIVCurve(year)
        self.bstring.getIVCurve(year)
        self.totalIndividualPower = self.fstring.totalIndividualPower + self.bstring.totalIndividualPower
        self.voltages = np.array(self.bstring.stringVoltages)
        self.currents = np.sum([np.array(self.fstring.stringCurrents), np.array(self.bstring.stringCurrents)], axis=0)
        self.powers = self.voltages * self.currents
        self.maxp = np.max(self.powers)
        
    def plotPowerCurve(self, directory, suptitle, title, time=None, axes=None, extra_text=None):
        zeros = np.where(self.powers == 0)[0]
        mismatch = 1.0 - self.maxp / self.totalIndividualPower
        first_zero = self.voltages[zeros[1]]
        if not axes:
            xlim = first_zero * 1.1
            ylim = self.maxp * 1.1        
        else:
            xlim = axes[0]
            ylim = axes[1]
        plt.plot(self.voltages[0:(zeros[1]+1)], self.powers[0:(zeros[1]+1)])
        plt.axis([0, xlim, 0, ylim])
        plt.suptitle(suptitle)
        plt.title(title)
        text_space = 0.04
        text_top = 0.95
        plt.text(xlim * 0.05, ylim * (text_top - 0 * text_space), "Time: {}".format(time), size=8)
        plt.text(xlim * 0.05, ylim * (text_top - 1 * text_space), "Maximum power: " + '{:6.2f}'.format(self.maxp) + " W", size=8)
        plt.text(xlim * 0.05, ylim * (text_top - 2 * text_space), "Mismatch losses: " + '{:.2f}'.format(mismatch * 100.0) + "%", size=8)
        plt.text(xlim * 0.05, ylim * (text_top - 3 * text_space), "Average Front Irrad.: " + '{:.2f}'.format(self.avg_front_irradiance) + " W/m2", size=8)
        plt.text(xlim * 0.05, ylim * (text_top - 4 * text_space), "Average Back Irrad.: " + '{:.2f}'.format(self.avg_back_irradiance) + " W/m2", size=8)
        for index,text in enumerate(extra_text):
            plt.text(xlim * 0.05, ylim * (text_top - (5 + index) * text_space), text, size=8)
        plt.xlabel("Module voltage (V)")
        plt.ylabel("Module power (W)")
        plt.savefig(directory + 'PowerCurve_' + str(time.year) + '_' + str(time.month) + '_' + str(time.day) + '_' + str(time.hour).zfill(2) + '_' + str(time.minute).zfill(2) + '.png')
        plt.close()
        returnArray = [time, self.maxp, self.totalIndividualPower, mismatch]
        returnArray.extend(self.irradString)
        return returnArray
            
        
   
def main():
    
    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    degradationParams = [0.998,0.0002,0.993,0.0002,1.01,0.1,0.975,0.015]
    env = Environment(1000,298.15)
    panel = Panel(params,env)                                        # Set the module params and the environment
    panel.setDegradationModel(degradationParams,20)                  # Set the degradation and pre-generate performance degradation over 20 years
    panel.generateIVCurve(20)                                        # Generate the IV curve for this module at year 20
    panel.plotIVCurve()                                              # Plot the curve
    
    '''
    #os.system(Model.RADBINS + 'gendaylit -ang 60 20 -W 700 300')
    os.system(Model.RADBINS + 'obj2rad ' + Model.RADBINS +'data/cube.obj > ' + Model.RADBINS +'data/cube.rad')
    
    params = [24.0, 15.2, 8.99, 0.1167, 1800.0, 1.956, 0.992, "Trina TSM-315PC14", 105.0, 3, -0.0032, 0.0005, 1.045]
    manufacturingParams = [1.0, 0.0072, 1.0, 0.0134]  # These ratios are rough estimates from the paper - Reis et al, 2002
    tolerances = [0.0, 5.0 / 315.0 * 100.0]  # Typical manufacturing tolerance 315 W (-0W/+5W)
    degradationParams = [0.998, 0.0002, 0.993, 0.0002, 1.01, 0.1, 0.975, 0.015]
    eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 0.0, 1100.0, 800.0, 0.02, 3.0)
    eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
    bf = BFModule(eg, params, manufacturingParams, tolerances, degradationParams)
    irrads = [1212.0, 1212.0, 1212.0, 532.0, 514.0, 337.0]
    bf.setIrradiances(irrads)
    bf.getIVCurve(0)
    bf.plotPowerCurve()
    '''

'''
    params = [24.0, 15.10, 9.26, 0.1167, 900.0, 1.956, 0.992, "Canadian Solar CS6U320P", 106.667, 3, 0.0046, -0.152, 1.026]
    degradationParams = [0.998, 0.0002, 0.993, 0.0002, 1.01, 0.1, 0.975, 0.015]
    manufacturingParams = [1.0, 0.0072, 1.0, 0.0134]  # These ratios are rough estimates from the paper - Reis et al, 2002
    tolerances = [0.0, 5.0 / 315.0 * 100.0]  # Typical manufacturing tolerance 315 W (-0W/+5W)
    eg = EnvironmentGenerator(298.15, 0.0, 1000.0, 40.0, 1100.0, 800.0, 0.02, 3.0)
    eg.setGenerationMode(eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE, eg.PARAMETER_INPUT_MODE) # Don't calculate inputs, they will be input
    PS = PanelString(eg, params, manufacturingParams, tolerances, 54)
    PS.setDegradationModel(degradationParams, 20)  # Generate a DegradationModel to generate and store all of the degradation charactersistics of the panels up to year 20
    PS.createPanels()  # Now use a PanelFactory to create all of the panels and assign them to this String
    PS.getIVCurve(0)  # Generate an IV curve for this string at year 10
    PS.plotPowerCurve()
    
'''    

if __name__ == "__main__":
    main()



