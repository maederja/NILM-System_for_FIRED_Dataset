

# various modules
import pickle
import numpy as np
import os
from datetime import datetime
import json
import helper as hp
import math

# for printing dicts in a more readable way
from beeprint import pp

# sorting array efficiently
from operator import itemgetter

#SKLEARN
from sklearn.model_selection import GridSearchCV

# Set FIRED base FIRED_BASE_FOLDER
hp.FIRED_BASE_FOLDER = "/media/jannik/TOSHIBAEXT/FIRED"

# choose which method for extracting the features was used
# choose between "cleaning" and "difference"
METHOD = "cleaning"
FILENAME_EVENTS = "eventFeatures_" + METHOD + "_final_splitted.pickle"
FILENAME_CLASSIFIERS = "trainedClassifiers_final_difference_noDBSCAN_21_06_12_07.pickle"

# chose if manufacturer information should be used for power usage or if the power actual
# power usage should be determined from the aggregated data.
USE_MANUFACTURER_INFORMATION = True


def loadEventsPickle(filename):
    filehandler = open(os.path.join("data", filename),"rb")
    file = pickle.load(filehandler)
    filehandler.close()
    x = file[1]

    return x


def loadClassifierPickle(filename):
    filehandler = open(os.path.join("data", filename),"rb")
    file = pickle.load(filehandler)
    filehandler.close()
    x = file[0]
    y = file[1]
    z = file[2]

    return x, y, z


def flatten(x_not_flat):
    X = []
    for x in x_not_flat:
        x_list = []
        for x_element in x:
            if isinstance(x_element,(list, np.ndarray)):
                x_list.extend(x_element)
            else:
                x_list.append(x_element)
        X.append(x_list)
    return X


def classify(features, classifiers, selectedFeatures, scaler):
    results = {"l1": {"up": {}, "down": {}}, "l2": {"up": {}, "down": {}}, "l3": {"up": {}, "down": {}}}

    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]:
            print("Working on {0} with direction {1}".format(phase, direction))
            # extract only the features the classifiers was trained on
            #print(features[phase][direction])
            features_sel = [[x[f] for f in selectedFeatures] for x in features[phase][direction]]
            features_flat = flatten(features_sel)
            features_scaled = scaler.transform(features_flat)

            for model in classifiers[phase][direction].keys():
                classifier = classifiers[phase][direction][model]["classifier"]
                print("{}...".format(model))
                label_pred = classifier.predict(features_scaled)
                results[phase][direction][model] = label_pred
    return results


def assignResults(features, results):
    assignedResults = {}
    for model in results["l3"]["up"].keys():
        assignedResults[model] = {}
        for phase in ["l1", "l2", "l3"]:
            events = []
            appliances = []
            for direction in ["up", "down"]:
                appliances_set = set(results[phase][direction][model])
                appliances.extend(appliances_set)
                timestamps = []
                direction_list = []
                for i in range(len(features[phase][direction])):
                    timestamps.append(features[phase][direction][i]["timestamp"])
                    direction_list.append(direction)
                events_temp = list(zip(timestamps, results[phase][direction][model], direction_list))
                events.extend(events_temp)
            appliances = set(appliances)
            events = sorted(events, key=itemgetter(0))
            for appliance in appliances:
                applianceEvents = [e for e in events if e[1] == appliance]
                assignedResults[model][appliance] = applianceEvents
    return assignedResults


def getONPhases(assignedResults):
    ONPhases = {}
    for model in assignedResults.keys():
        ONPhases[model] = {}
        for appliance in assignedResults[model].keys():
            ONPhases[model][appliance] = []
            for i in range(len(assignedResults[model][appliance])-1):
                # check if ith element has label "on" and (i+1)th element has label "down"
                if assignedResults[model][appliance][i][2] == 'up' and assignedResults[model][appliance][i+1][2] == 'down':
                    # calculate the duration of the event
                    difference = datetime.timestamp(assignedResults[model][appliance][i+1][0]) - datetime.timestamp(assignedResults[model][appliance][i][0])
                    ONPhases[model][appliance].append((assignedResults[model][appliance][i][0], assignedResults[model][appliance][i+1][0], difference))
    return ONPhases


def estimatePowerUsage(ONPhases):
    appliancePowerUsage = {}
    for model in ONPhases.keys():
        appliancePowerUsage[model] = {}
        for appliance in ONPhases[model].keys():
            appliancePowerUsage[model][appliance] = {"phases": [], "completePowerUsage": 0}
            powerRating = 0
            # Uses the estimated power usage given by the manufacturer
            if USE_MANUFACTURER_INFORMATION == True:
                powerRatingData = hp.getDeviceInfo()
                # only powerRating of appliances is given so we have to look up which appliances are on which powermeter
                deviceMapping = hp.getDeviceMapping()

                if appliance in deviceMapping.keys():
                    appliances_powermeter = deviceMapping[appliance]["appliances"]
                else:
                    appliances_powermeter = [appliance]

                #check if there are multiple devices on the powermeter
                if len(appliances_powermeter) > 1:
                    sum = 0
                    i = 0
                    for appliance_powermeter in appliances_powermeter:
                        i += 1
                        sum += powerRatingData[appliance_powermeter]["powerRating"]
                    powerRating = sum / i
                else:
                    appliance_powermeter = appliances_powermeter[0]
                    # coffee grinder and espresso machine have no powerRating in deviceInfo.json
                    if appliance_powermeter == "coffee grinder":
                        powerRating = 128
                    elif appliance_powermeter == "espresso machine":
                        powerRating = 1200
                    elif appliance_powermeter == "NULL":
                        powerRating = 0
                    else:
                        powerRating = powerRatingData[appliance_powermeter]["powerRating"]

            completePowerUsage = 0
            for ONPhase in ONPhases[model][appliance]:
                # Uses the power usage difference before and after the UP-event
                if USE_MANUFACTURER_INFORMATION == False:
                    phase = hp.getPhase(appliance)
                    power = "p"
                    startTs = datetime.timestamp(ONPhase[0])
                    aggregatedData = hp.getMeterPower("smartmeter001", 1, startTs - 10, startTs + 10)
                    data = {k:v for k, v in aggregatedData.items() if k != "data"}
                    data["data"]= aggregatedData["data"][[name for name in aggregatedData["data"].dtype.names if name in [str(power) + "_l1", str(power) + "_l2", str(power) + "_l3"]]]
                    dataBefore = data["data"]["{0}_l{1}".format(power, phase)][0:10]
                    dataAfter = data["data"]["{0}_l{1}".format(power, phase)][10:20]
                    dataBeforeMean = np.mean(dataBefore)
                    # TODO: was bedeutet arg?
                    dataAfterMean = np.mean(dataAfter)
                    difference = dataAfterMean - dataBeforeMean
                    powerRating = difference

                # CALCULATE the power usage after determining the powerRating of the different appliances
                duration = ONPhase[2]
                # need hours to calculate the power usage in kWh
                duration_hour = duration / 60 / 60
                # devide by 1000 to get from Wh to kWh
                powerUsage = duration_hour * powerRating / 1000
                appliancePowerUsage[model][appliance]["phases"].append({"startTs": ONPhase[0], "stopTs": ONPhase[1], "duration": ONPhase[2], "powerUsage": powerUsage})
                completePowerUsage += powerUsage

            appliancePowerUsage[model][appliance]["completePowerUsage"] = completePowerUsage

    return appliancePowerUsage


# Problem hier: Wenn Event zum Start aktiv war, dann werden alle Daten falsch
def getPowerUsage(assignedResults):
    appliancePowerUsage = {}
    for model in assignedResults.keys():
        appliancePowerUsage[model] = {}
        for appliance in assignedResults[model].keys():
            appliancePowerUsage[model][appliance] = {"complete" : 0, "events" : []}
            phase = hp.getPhase(appliance)
            if phase == -1: continue
            power = "p"
            # add 10s to start ands top time, because we need a 10s range around every event
            startTs = datetime.timestamp(assignedResults[model][appliance][0][0]) - 10
            stopTs = datetime.timestamp(assignedResults[model][appliance][-1][0]) + 10
            aggregatedData = hp.getMeterPower("smartmeter001", 1, startTs, stopTs)
            data = {k:v for k, v in aggregatedData.items() if k != "data"}
            data["data"]= aggregatedData["data"][[name for name in aggregatedData["data"].dtype.names if name in [str(power) + "_l" + str(phase)]]]
            currentPower = 0
            powerUsageComplete = 0

            for i in range(len(assignedResults[model][appliance]) - 1):
                event = assignedResults[model][appliance][i]
                eventTs = datetime.timestamp(event[0])
                index = round(eventTs - startTs) + 1
                dataBefore = data["data"]["{0}_l{1}".format(power, phase)][index-10:index]
                dataAfter = data["data"]["{0}_l{1}".format(power, phase)][index:index+10]
                dataBeforeMean = np.mean(dataBefore)
                dataAfterMean = np.mean(dataAfter)
                powerDifference = dataAfterMean - dataBeforeMean
                currentPower += powerDifference
                duration = datetime.timestamp(assignedResults[model][appliance][i+1][0]) - eventTs
                # convert to hours and kWh
                powerUsage = (((duration * currentPower) / 60 ) / 60 ) / 1000
                powerUsageComplete += powerUsage

                eventData = {"timestamp" : eventTs, "duration" : duration, "powerDifference" : powerDifference, "powerUsage" : powerUsage, "currentPower" : currentPower, "direction": event[2]}
                appliancePowerUsage[model][appliance]["events"].append(eventData)

            appliancePowerUsage[model][appliance]["complete"] = powerUsageComplete
    return appliancePowerUsage


print("Loading events and the trained classifiers for Classification...")
features = loadEventsPickle(FILENAME_EVENTS)
classifiers, selectedFeatures, scaler = loadClassifierPickle(FILENAME_CLASSIFIERS)

print("Predicting the appliance for all given events...")
results = classify(features, classifiers, selectedFeatures, scaler)

print("Assigning the predicted appliances to the event timestamps...")
assignedResults = assignResults(features, results)

print("Estimating the power usage based on the events of the appliances...")
appliancePowerUsage = getPowerUsage(assignedResults)

print("Estimating the power usage based on looking for the On-phases of the appliances and using manufacturer information for power usage")
OnPhases = getONPhases(assignedResults)
appliancePowerUsageOnPhases = estimatePowerUsage(OnPhases)
