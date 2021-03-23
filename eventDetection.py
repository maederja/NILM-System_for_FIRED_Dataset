#TODO:
# Eventuell müssen die Thresholds der Powermeters nochmal angepasst werden
# für Powermeter wurde bei getEventsFromData eine Option ergänzt, dass indices weiter eingegränz werden
# --> Option arbeitet zu aggressiv. Hier prüfen ob es besser ist immer die Option
#     die prüft ob Werte fur bestimmten Zeitraum über Threshold bleiben einzugliedern
# changed the detection range for lamps from 1s to 5s. Otherwise lamps won't be found



# import necessary modules
import helper as hp
import numpy as np
from datetime import datetime
from scipy import signal
import math
import pickle
import os

# for printing complete numpy arrays
np.set_printoptions(suppress=True, threshold=np.inf)

# SETUP:
# Set FIRED base FIRED_BASE_FOLDER
hp.FIRED_BASE_FOLDER = "/media/jannik/TOSHIBAEXT/FIRED"

# Set which data should be used
#recordingRange = hp.getRecordingRange("2020.06.14", "2020.09.23")
recordingRange = hp.getRecordingRange("2020.06.21", "2020.07.12")
# Set power type (p, q, s)
powerType = "s"

NUM_SINE_FROM_BEFORE = 4
LOAD_BEFORE_EVENT = 3.0 #changed from 1.5
LOAD_AFTER_EVENT = 3.0
TIME_BEFORE_EVENT = 0.5 #changed from 0.5
TIME_AFTER_EVENT = 1.5 #changed from 1.0


# set thresholds for different powermeters
# IDLE is no longer used!!!!!
eventThresholds = {
    "smartmeter001": {
        "l1": {"up": 6 , "down": -6 , "idle": 90 },  # "l1": {"up": 10 , "down": -7 , "idle": 90 }
        "l2": {"up": 6 , "down": -6 , "idle": 60 }, # "l2": {"up": 10 , "down": -10 , "idle": 60 }
        "l3": {"up": 6 , "down": -6 , "idle": 72 }, # "l3": {"up": 30 , "down": -30 , "idle": 72 }
    },
    "devices": {
        "powermeter08": {"up":30  ,"down":-30  ,"idle":5  },
        "powermeter09": {"up":50  ,"down":-50  ,"idle":5  }, #fridge -> no door open
        "powermeter10": {"up":10  ,"down":-10  ,"idle":1  },
        "powermeter11": {"up":4   ,"down":-4   ,"idle":1  },
        "powermeter12": {"up":8   ,"down":-4   ,"idle":2  },
        "powermeter13": {"up":50  ,"down":-50  ,"idle":5  },
        "powermeter14": {"up":5   ,"down":-5   ,"idle":5  }, #should be something like 0.5 --> events will never be detected
        "powermeter15": {"up":100 ,"down":-100 ,"idle":2  },
        "powermeter16": {"up":100 ,"down":-100 ,"idle":5  },
        "powermeter17": {"up":100 ,"down":-100 ,"idle":5  },
        "powermeter18": {"up":5   ,"down":-5   ,"idle":5  }, #changed from 20
        "powermeter19": {"up":5   ,"down":-5   ,"idle":5  }, #changed from 20
        "powermeter20": {"up":5   ,"down":-5   ,"idle":5  }, #changed from 30
        "powermeter21": {"up":5   ,"down":-5   ,"idle":5  }, #changed from 20
        "powermeter22": {"up":30  ,"down":-30  ,"idle":28 },
        "powermeter23": {"up":50  ,"down":-50  ,"idle":12 },
        "powermeter24": {"up":10  ,"down":-10  ,"idle":9  },
        "powermeter25": {"up":15  ,"down":-10  ,"idle":5  },
        "powermeter26": {"up":5   ,"down":-5   ,"idle":5  }, #should be something like 1.0 --> events will never be detected
        "powermeter27": {"up":5   ,"down":-5   ,"idle":5  }, #changed from 20
        "powermeter28": {"up":10  ,"down":-10  ,"idle":5  },
        # Special devices of powermeter11
        "battery charger":  {"up":5  ,"down":-4,  "idle":1},
        "toothbrush":       {"up":5  ,"down":-4,  "idle":1},
        "laptop #1":        {"up":5 ,"down":-5, "idle":2},
        "razor":            {"up":5  ,"down":-4,  "idle":2},
        "laptop #2":        {"up":5 ,"down":-5, "idle":2},
        "kitchen machine":  {"up":8  ,"down":-4,  "idle":2},
        "sewing machine":   {"up":8  ,"down":-4,  "idle":2},
        #stove
        "stove":            {"up": 100, "down": -100, "idle": 5},
    }
} # IDLE is no longer used!!!!!


def extractDataPhase(data, power):
    # Copys array for further analysis
    data = {k:v for k, v in aggregatedData.items() if k != "data"}
    # Extract only one power-type

    data["data"]= aggregatedData["data"][[name for name in aggregatedData["data"].dtype.names if name in [str(power) + "_l1", str(power) + "_l2", str(power) + "_l3"]]]
    # Convert dtype name from "p_l1", etc to "l1"
    data["data"].dtype.names = data["measures"] = [n.split("s_")[1] for n in data["data"].dtype.names]
    return data


def getEventsFromData(data, samplingrate, timestamp, thresUp, thresDown, thresIdle, powermeter=False, smartmeter=False, stove=False):
    M = int(5.0*samplingrate)
    # Moving average over 3 seconds
    # Sum up the differences in the last N seconds
    N = int(3.0*samplingrate)
    # Makes sharper edges
    diffData = np.diff(signal.medfilt(data, N))
    diffData = np.convolve(diffData, np.ones(N, dtype=int), 'full')

    events= {"up":[], "down":[]}
    for direction in ["up", "down"]:
        if direction == "up":
            indices = np.where(diffData > thresUp)[0]
            # Discard events caused by Fired meter dropout
            if stove==False:
                indices = [e for e in indices if all(data[e-i] != 0.0 for i in range(1,M) if e-i >= 0)]
            # Must be above idle at least M seconds after and below before
            # throws out events that don't match the requirements
            # not possible when using aggregated data, because events can stack
            if powermeter==True: #don't use. this will only sorts out events that have been matched already
                #indices = [e for e in indices if any(data[e-i] < thresIdle for i in range(1,M) if e-i >= 0)] # too agressive (multi-step appliances will not be recognized)
                indices = [e for e in indices if all(data[e+i] > thresIdle for i in range(1,M) if e+i < len(data))] #can be included
            if smartmeter==True:
                indices = [e for e in indices if all(data[e+i] > np.mean(data[e-int(samplingrate):e]) for i in range(1,M) if e+i < len(data))]
            # Keep first one (of the same events detected) if they are within 5s
            f = np.where(np.diff(np.array([0]+list(indices))) > 5*samplingrate)[0]
            indices = [ indices[i] for i in f ]
            #print(indices)

        elif direction == "down":
            indices = np.where(diffData < thresDown)[0]
            # Discard events caused by Fired meter dropout
            if stove==False:
                indices = [e for e in indices if all(data[e+i] != 0.0 for i in range(1,M) if e+i < len(data))]
            # Must be above idle at least M seconds before and below after
            # not possible when using aggregated data, because events can stack
            if powermeter==True: #don't use. this will only sorts out events that have been matched already
                #indices = [e for e in indices if any(data[e+i] < thresIdle for i in range(1,M) if e+i < len(data))] (multi-step appliances will not be recognized)
                indices = [e for e in indices if all(data[e-i] > thresIdle for i in range(1,M) if e-i >= 0)] #can be included
            if smartmeter==True:
                indices = [e for e in indices if all(data[e-i] > np.mean(data[e:e+int(samplingrate)]) for i in range(1,M) if e-i <=0)]
            if len(indices) < 1: continue
            # Keep last one if they are within 5s
            f = np.where(np.diff(np.array([indices[-1] + 10]+list(reversed(list(indices))))) < -5*samplingrate)[0]
            indices = [ list(reversed(indices))[i] for i in f ]
            indices = sorted(indices)
            #print(indices)

        # Covert index to timestamp
        ts = timestamp
        tstep = 1.0/samplingrate
        times = [ts + e*tstep for e in indices]
        difference = [diffData[e] for e in indices]
        events[direction] = [(e) for e in zip(times, difference)]
    return events


def getEventsFromAggregatedData(data, samplingrate, timestamp, thresholds):
    events = { "l1": {}, "l2": {}, "l3": {}}
    for phase in ["l1", "l2", "l3"]:
        dataPhase = data[phase]
        events[phase] = getEventsFromData(data[phase], samplingrate, timestamp, thresholds[phase]["up"], thresholds[phase]["down"], thresholds[phase]["idle"], powermeter=False, smartmeter=True)
    return events


def checkApplianceForEvent(events, samplingrate, startTs, stopTs, thresholds, power):
    assignedEvents = {"l1": {"up": [], "down": []}, "l2": {"up": [], "down": []}, "l3": {"up": [], "down": []}}
    #load stove data for all phases
    stoveData = hp.getPowerStove(1, startTs=startTs, stopTs=stopTs)
    count = len(events["l1"]["up"]) + len(events["l1"]["down"]) + len(events["l2"]["up"]) + len(events["l2"]["down"]) + len(events["l3"]["up"]) + len(events["l3"]["down"])
    print(count)
    j=-1
    k=0
    for phase in [1, 2, 3]:
        # Get all meters on this phase
        metersOnPhase = [m for m in hp.getMeterList() if hp.getPhase(m) == phase]
        # get all lights
        lights = [a for a in hp.getApplianceList() if "light" in a]
        # only light on this phase
        lightsOnPhase = [m for m in lights if hp.getPhase(m) == phase]

        #extract stove data for the current phase
        stoveDataPhase = stoveData[phase - 1]
        #extract the events from stove data
        stoveEvents = getEventsFromData(stoveDataPhase["data"][power], stoveDataPhase["samplingrate"], stoveDataPhase["timestamp"], thresholds["stove"]["up"], thresholds["stove"]["down"], thresholds["stove"]["idle"], powermeter=False, smartmeter=False, stove=True)
        meterEvents = {}
        assignDevice = ""
        for meter in metersOnPhase:
            meterData = hp.getMeterPower(meter, 1, startTs, stopTs)
            meterEvents[meter] = getEventsFromData(meterData["data"][power], meterData["samplingrate"], meterData["timestamp"], thresholds[meter]["up"], thresholds[meter]["down"], thresholds[meter]["idle"], powermeter=False, smartmeter=False)
        for direction in ["up", "down"]:
            for event in events["l{0}".format(phase)][direction]:

                diff = event[1]
                event = event[0]

                #just for showing the progress of the process
                j += 1
                if (j%100) == 0:
                    print("{0}/{1}".format(j, count))

                # if this is applied, a NULL-Class with all events that couldn't be matched, will be introduced
                assignDevice = "NULL"
                eventTime = 0

                i = 0
                # NOTE: when using 3, an actual range of +/-2 is used (because of the sampling rate and usage of < instead of <=)
                rangeMeter = 3
                # light events seem to be registered +/- 5s around the actual event
                rangeLight = 5

                # powermeter
                for meter in metersOnPhase:
                    for meterEvent in meterEvents[meter][direction]:
                        if (i > 1): break
                        if abs(meterEvent[0] - event) < rangeMeter:
                            assignDevice = meter
                            eventTime = meterEvent[0]
                            i += 1

                # stove
                for stoveEvent in stoveEvents[direction]:
                    if (i>1): break
                    if abs(stoveEvent[0] - event) < rangeMeter:
                        assignDevice = "stove"
                        eventTime = stoveEvent[0]
                        i += 1

                # lights
                for light in lightsOnPhase:
                    if i > 1: continue
                    lightData = next(l for l in hp.loadAnnotations(hp.LIGHT_ANNOTATION, loadData=False) if l["name"] == light)
                    lightData["data"] = hp.convertToTimeRange(hp.loadCSV(lightData["file"]), clipLonger=12*60*60, clipTo=10*60)
                    for entry in lightData["data"]:
                        eventBefore = event - rangeLight
                        eventAfter = event + rangeLight
                        if entry["startTs"] > eventAfter or entry["stopTs"] < eventAfter: continue
                        if direction == "up":
                            if "state" not in entry or entry["state"].lower() == "off": continue
                            if entry["startTs"] > eventBefore and entry["startTs"] < eventAfter:
                                i += 1
                                assignDevice = light
                        if direction == "down":
                            if "state" not in entry or entry["state"].lower() == "on": continue
                            if entry["startTs"] > eventBefore and entry["startTs"] < eventAfter:
                                i += 1
                                assignDevice = light


                # if powermeter11 was active, check which device it was
                if assignDevice == "powermeter11" and i==1:
                    for entry in hp.getChangingDeviceInfo():
                        if direction == "up":
                            if entry["startTs"] < event and entry["stopTs"] > event:
                                assignDevice = entry["name"]
                        if direction == "down":
                            if entry["startTs"] < event and entry["stopTs"] > event:
                                assignDevice = entry["name"]

                if i > 1:
                    k += 1

                # assign device to timestamp
                if i==1 or assignDevice == "NULL":
                    # skip unmatched events which have a power-jump smaller than 8
                    if abs(diff) <= 10 and assignDevice == "NULL": continue
                    #if assignDevice == "NULL":
                    #    print(assignDevice, datetime.fromtimestamp(event), phase, direction, diff)
                    assignedEvents["l{0}".format(phase)][direction].append((event, assignDevice))
    print("events skipped because multiple matches were found: {}".format(k))
    return assignedEvents


# no longer used
def calcPowers(voltage, current, samplingRate, upsample=False, LINE_FREQUENCY=50.0):
    """
    Calculate Active, Reactive and Apparent power from voltage and current.

    :param voltage:      Voltage in volt
    :type  voltage:      list or np.array
    :param current:      Current in milli ampere
    :type  current:      list or np.array
    :param samplingRate: Samplingrate for phase calculation
    :type  samplingRate: int
    :param upsample: If final data should be same samplingrate as input data, default=False
                     If set to false, the returned power is 50Hz
    :type  upsample: bool
    :return: Active, Reactive and Apparent Power as np.arrays
    :rtype: Tuple
    """

    sfos = int(samplingRate/LINE_FREQUENCY)
    numPoints = len(voltage)
    reshaping = int(math.floor(numPoints/sfos))
    end = reshaping*sfos
    # Make both mean free
    v = voltage[:end]
    c = current[:end]
    momentary = 0.001*np.array(v[:end]*c[:end])
    # bringing down to 50 Hz by using mean
    momentary = momentary.reshape((-1, sfos))
    p = np.mean(momentary, axis=1)
    v = v[:end].reshape((-1, sfos))
    i = c[:end].reshape((-1, sfos))
    # quicker way to get rms than using dot product
    vrms = np.sqrt(np.einsum('ij,ij->i', v, v)/sfos)
    irms = np.sqrt(np.einsum('ij,ij->i', i, i)/sfos)
    # Because unit of current is in mA
    s = 0.001*vrms*irms
    q = np.sqrt(np.abs(s*s - p*p))
    return p,q,s


def nextPeriodStart(v, sr):
    """ Extract the index a new period starts. """
    sfos = int(math.ceil(sr/50))
    # Find point where voltage goes up
    e = min(3*sfos, len(v))
    signs = np.sign(v[:e])
    # Since np.sign(0) yields 0, we treat them as negative sign here
    signs[signs == 0] = -1
    zc = np.where(np.diff(signs))[0]
    deltaIndex = -1
    if len(zc) > 1:
        # either the first or 2nd zerocrossing is the voltage going up
        if v[zc[0]+1] > 0: return zc[0]
        elif v[zc[1]+1] > 0: return zc[1]
    # We should not end here
    # assert AssertionError("Cannot find period index")
    return -1


def extractROIindex(dataDict, phase, direction, roughJump:float=None, powerJump:float=5.0, LINE_FREQ:float=50, verbose=False):
    # faster access
    sr = dataDict["samplingrate"]
    data = dataDict["data"]

    sfos = int(sr/LINE_FREQ)
    # maxlen that can be extracted from this
    maxLen = int(len(data)/sfos)*sfos
    # num phases inside data
    phases = int(maxLen/sfos)
    # lets calculate apparent power from the jump
    # NOTE: maybe its quicker to use RMS current for this
    # quicker way to do this
    if direction == "down":
        # reverse the data array
        dataV = data["v_{}".format(phase)][::-1]
        dataI = data["i_{}".format(phase)][::-1]
    elif direction == "up":
        dataV = data["v_{}".format(phase)]
        dataI = data["i_{}".format(phase)]
    v = dataV[:maxLen].reshape((-1, sfos))
    i = dataI[:maxLen].reshape((-1, sfos))
    vrms = np.sqrt(np.einsum('ij,ij->i', v, v)/sfos)
    irms = np.sqrt(np.einsum('ij,ij->i', i, i)/sfos)
    # Because unit of current is in mA
    s = 0.001*vrms*irms
    # no rough estimation given, so use data mid
    if roughJump is None: roughJump = (len(data)/sr)/2.0
    # convert to index in power data
    roughI = max(0, min(len(s), int(roughJump*LINE_FREQ)))

    if verbose: print("roughJump: {}, roughI: {}".format(roughJump, roughI))
    # get index of minimum apparent power in first data half
    minI = np.argmin(s[:roughI])
    if verbose: print("minI: {}".format(minI))
    thres = np.mean(s[max(0,minI-3):minI+1])
    if verbose: print("Thres: {}W".format(thres))
    index = -1
    # thres += max(thres*0.1, powerJump)
    thres = max(thres*0.1, powerJump) # we only look at diff signal
    if verbose: print("Thres: {}W".format(thres))

    indices = np.where(np.diff(s) > thres)[0]
    #print(indices)
    if len(indices) > 0:
        index = indices[(0.5*np.abs(indices - roughI) + 0.5*1.0/np.diff(s)[indices]).argmin()]
    if verbose: print(index)
    if index > -1:
        index *= sfos
        if direction == "down":
            index = len(dataV) - index
        index += nextPeriodStart(data["v_{}".format(phase)][index:], sr)
    return index

# Calculate the exact timestamp using voltage and current
def getExactIndex(events, threshold):
    x = {"l1": { "up": [], "down": [] }, "l2": { "up": [], "down": [] }, "l3": { "up": [], "down": [] }}
    y = {"l1": { "up": [], "down": [] }, "l2": { "up": [], "down": [] }, "l3": { "up": [], "down": [] }}
    length = 0
    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]:
            length = length + len(events[phase][direction])
    i = -1
    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]:
            for event in events[phase][direction]:
                i += 1
                if (i%10) == 0:
                    print("{0}/{1}".format(i, length))
                ts = event[0]
                appliance = event[1]
                dataDict = hp.getMeterVI("smartmeter001", startTs=ts-LOAD_BEFORE_EVENT, stopTs=ts+LOAD_AFTER_EVENT)
                if dataDict is None: continue
                index = extractROIindex(dataDict, phase, direction)
                if index <= -1: continue

                """
                mi = max(0, index - int(TIME_BEFORE_EVENT * dataDict["samplingrate"]))
                ma = min(len(dataDict["data"]), index + int(TIME_AFTER_EVENT * dataDict["samplingrate"]))
                #print(mi, ma)
                # only save data of the current phase
                dataNew = dataDict["data"][[name for name in dataDict["data"].dtype.names if name in ["v_" + str(phase), "i_" + str(phase)]]]
                dataNew.dtype.names = [n.split("_")[0] for n in dataNew.dtype.names]
                dataNew = dataNew[mi:ma]
                index_adjust = index - mi
                """

                nTs = dataDict["timestamp"] + index/dataDict["samplingrate"]
                tsBefore = nTs - TIME_BEFORE_EVENT
                tsAfter = nTs + TIME_AFTER_EVENT

                del dataDict

                x[phase][direction].append((nTs, appliance, tsBefore, tsAfter))
    return x


def saveDataPickle(x):
    #filename = "FIREDEventsExtracted_" + datetime.fromtimestamp(recordingRange[0]).strftime("%d_%m_%Y") + "-" + datetime.fromtimestamp(recordingRange[1]).strftime("%d_%m_%Y") + ".pickle"
    filename = "FIREDEventsExtracted_final_21_06_12_07.pickle"
    filehandler = open(os.path.join("data", filename),"wb")
    pickle.dump(x, filehandler)
    filehandler.close()
    return



# EXTRACTION FROM AGGREGATED DATA
print("Getting aggregated power data...")
aggregatedData = hp.getMeterPower("smartmeter001", 1, recordingRange[0], recordingRange[1])

print("Analyzing aggregated power data...")
dataPowerType = extractDataPhase(aggregatedData, powerType)
meter = "smartmeter001"
events = getEventsFromAggregatedData(dataPowerType["data"], dataPowerType["samplingrate"], dataPowerType["timestamp"], eventThresholds[meter])



print("Extracted all events from aggregated power data")
print("Trying to match events with data from individual devices...")
assignedEvents = checkApplianceForEvent(events, dataPowerType["samplingrate"], recordingRange[0], recordingRange[1], eventThresholds["devices"], powerType)


print("Calculating the exact timestamp of the matched events...")
# x is the event timestamp and the matching appliance: y[phase][direction][event]
x = getExactIndex(assignedEvents, eventThresholds[meter])



print("Extracted the exact timestamp of the all events")
print("Storing event timestamps and data before and after the event in a pickle file...")
saveDataPickle(x)

print("Data stored successfully")
