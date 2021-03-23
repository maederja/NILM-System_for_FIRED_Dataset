# for printing dicts in a more readable way
from beeprint import pp

# SKLEARN for clustering the events of the null-class
from sklearn.cluster import DBSCAN

import pickle
import helper as hp
import math
import numpy as np
from datetime import datetime
import scipy
import os


np.set_printoptions(suppress=True, threshold=np.inf)

hp.FIRED_BASE_FOLDER = "/media/jannik/TOSHIBAEXT/FIRED"
FILENAME = "FIREDEventsExtracted_final_21_06_12_07.pickle"
LINE_FREQUENCY = 50
DELETE_EVENTS_WITHOUT_CLUSTER = True
CLUSTER_NULL_EVENTS = True


def loadDataPickle():
    filehandler = open(os.path.join("data", FILENAME),"rb")
    file = pickle.load(filehandler)
    filehandler.close()

    x = file

    return x


def rms(data):
    result = np.sqrt(data.dot(data)/data.size)
    return result


def getData(x, phase):
    ts = x[0]
    tsBefore = x[2]
    tsAfter = x[3]

    # load 0.5s before and 1.5s after the event
    dataDict = hp.getMeterVI("smartmeter001", startTs=tsBefore, stopTs=tsAfter)
    if dataDict is None: return

    index = int((ts - dataDict["timestamp"]) * dataDict["samplingrate"])

    dataNew = dataDict["data"][[name for name in dataDict["data"].dtype.names if name in ["v_" + str(phase), "i_" + str(phase)]]]
    dataNew.dtype.names = [n.split("_")[0] for n in dataNew.dtype.names]

    return dataNew, index


def fft(data, sr, win=None, scale="mag", beta=9.0):
    # Construct window
    win_data = np.ones(len(data))
    if win == "hamming": win_data = np.hamming(len(data))
    elif win == "hanning": win_data = np.hanning(len(data))
    elif win == "blackman": win_data = np.blackman(len(data))
    elif win == "kaiser": win_data = np.kaiser(len(data), beta)
    # Apply windowed FFT
    fft = np.abs(np.fft.rfft(win_data*data))/len(data)
    # Scale
    if scale == "mag": fft = fft*2
    elif scale == "pwr": fft = fft**2
    # Calc freqs
    xf = np.fft.rfftfreq(len(data), 1.0/sr)
    return fft, xf


# x is the event timestamp and the matching appliance: y[phase][direction][event]

def extractFeatures(x, array=True):
    eventFeatures = {"l1": {"up": [], "down": []}, "l2": {"up": [], "down": []}, "l3": {"up": [], "down": []}}
    eventAppliance = {"l1": {"up": [], "down": []}, "l2": {"up": [], "down": []}, "l3": {"up": [], "down": []}}
    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]: # down excluded because those values are not yet correct
            for i in range(0, len(x[phase][direction])):
                timestamp = x[phase][direction][i][0]
                appliance = x[phase][direction][i][1]

                x_data, index = getData(x[phase][direction][i], phase)
                # data has length of 2s
                sr = len(x_data)/2.0 # changed from 1.5

                """
                # Estimate Line Frequency by number of voltage zero crossings
                signs = np.sign(x_data[index:]["v"])
                # Since np.sign(0) yields 0, we treat them as negative sign here
                signs[signs == 0] = -1
                zc = len(np.where(np.diff(signs))[0])/2
                lineFreq = 60
                if abs(zc-50.0) < abs(zc-60.0): lineFreq = 50
                """

                lineFreq = 50

                sfos = int(math.ceil(sr/lineFreq))

                # AFTER
                dataAfter = x_data[index:]
                dataBefore = x_data[:index]
                # We want a full number of sines
                eAfter = int(int(len(dataAfter)/sfos))*sfos
                eBefore = int(int(len(dataBefore)/sfos))*sfos

                e = min(eBefore, eAfter)

                dataAfter = dataAfter[:eAfter]
                dataBefore = dataBefore[-eBefore:]

                if len(dataAfter) == 0 or len(dataBefore) == 0:
                    print("Event of {} got skipped, because the array length was 0".format(appliance))
                    continue

                data = np.array(dataAfter[:e]["i"] - dataBefore[-e:]["i"])


                #print(len(x_data), index, eBefore, eAfter)

                #*************************FEATURES******************************
                #***********using the feature additive criterion****************

                #*********Active power difference (P)***************************
                momentaryPowerBefore = 0.001*np.array(dataBefore["v"]*dataBefore["i"])
                momentaryPowerAfter = 0.001*np.array(dataAfter["v"]*dataAfter["i"])
                pBefore = np.mean(momentaryPowerBefore)
                pAfter = np.mean(momentaryPowerAfter)
                pDiff = pAfter - pBefore

                #pDiff2 = np.mean(momentaryPowerAfter - pBefore)


                #*********Apparent power difference (S)*************************
                vrmsBefore = rms(dataBefore["v"])
                irmsBefore = rms(dataBefore["i"])
                vrmsAfter = rms(dataAfter["v"])
                irmsAfter = rms(dataAfter["i"])

                sBefore = 0.001 * vrmsBefore * irmsBefore
                sAfter = 0.001 * vrmsAfter * irmsAfter
                sDiff = sAfter - sBefore


                #**********Reactive power difference (Q)************************
                qBefore = np.sqrt(np.abs(sBefore*sBefore - pBefore*pBefore))
                qAfter = np.sqrt(np.abs(sAfter*sAfter - pAfter*pAfter))
                qDiff = qAfter - qBefore


                #**********Resistance and Admittance****************************
                # correct way is 1/R = 1/R1 + 1/R2
                if vrmsBefore == 0 or irmsBefore == 0 or vrmsAfter == 0 or irmsAfter == 0:
                    print("Division by 0")
                    continue
                RBefore = vrmsBefore / irmsBefore
                RAfter = vrmsAfter / irmsAfter
                RDiff = 1.0/(1.0/RAfter - 1.0/RBefore)
                YDiff = 1.0/RDiff


                # exact the same results as calculation above
                """
                R = (vrmsAfter) / (irmsAfter - irmsBefore)
                Y = 1.0/R
                """

                #**********Crest Factor*****************************************
                # substracting current data before calculation step
                cf_sub = max(abs(data))/rms(data)

                # substracting after calculation step
                cfBefore = max(abs(dataBefore["i"]))/rms(dataBefore["i"])
                cfAfter = max(abs(dataAfter["i"]))/rms(dataAfter["i"])
                cfDiff = cfAfter - cfBefore

                #***********Form Factor*****************************************
                # distinguish switching power supplies from other loads
                FFBefore = rms(dataBefore["i"])/ np.mean(np.abs(dataBefore["i"]))
                FFAfter = rms(dataAfter["i"])/ np.mean(np.abs(dataAfter["i"]))
                FFDiff = FFAfter - FFBefore

                FF_sub = rms(data)/ np.mean(np.abs(data))


                #**********Log Attack Time**************************************
                # Time until maxiumum current is drawn. Appliances like power drills
                # with internal speed controll will show larger values here
                # This is only interesting for rush in data
                ######DATA is difference between dataAfter and dataBefore#######
                periods = [data[i*sfos:(i+1)*sfos] for i in range(int(len(data)/sfos))]
                cleanPeriods = periods[1:] # first period has startup peaks
                curOfPeriod = [rms(period) for period in periods]
                normCurOfPeriod = list(curOfPeriod/max(curOfPeriod))
                # + 1 to accomodate log(0)
                LAT = np.log(normCurOfPeriod.index(max(normCurOfPeriod))*20.0 + 1)


                #**********Temporal Centroid************************************
                # Temporal balancing point of current energy
                TC = 0
                for i, cur in enumerate(curOfPeriod):
                    TC += (i+1)*cur
                TC /= sum(curOfPeriod)*sr


                #**********Ratio Of Positive & Negative Half Cycles*************
                # Some appliances with e.g. dimmers or speed controllers show different
                # behavior in the positive compared to the negative halfcycle. An
                # Average over multiple halfcycles is taken and the rms of both are
                # compared
                posHal = sum([period[0:int(sfos/2)] for period in cleanPeriods]) / len(cleanPeriods)
                negHal = sum([period[int(sfos/2):] for period in cleanPeriods]) / len(cleanPeriods)
                rmsPosHal = rms(posHal)
                rmsNegHal = rms(negHal)
                if rmsPosHal >= rmsNegHal: PNR = rmsNegHal/rmsPosHal
                else: PNR = rmsPosHal/rmsNegHal


                #**********Max-Min Ratio****************************************
                # Alternative to express one sided waveform characteristics
                minI = abs(min(data))
                maxI = abs(max(data))
                if maxI >= minI: MAMI = minI/maxI
                else: MAMI = maxI/minI

                minIBefore = abs(min(dataBefore["i"]))
                minIAfter = abs(min(dataAfter["i"]))
                maxIBefore = abs(max(dataBefore["i"]))
                maxIAfter = abs(max(dataAfter["i"]))
                maxI = maxIAfter - maxIBefore
                minI = minIAfter - minIBefore
                if maxI >= minI: MAMI_Diff = minI/maxI
                else: MAMI_Diff = maxI/minI


                #**********Peak-Mean Ratio**************************************
                # Determine if appliance has pure sine current of spikes from
                # switching artifacts
                # only take dataAfter --> really unlikely that in the 1.5s timespan
                # another device could be activated which changes ratio
                PMR = max(abs(dataAfter["i"])) / np.mean(abs(dataAfter["i"]))
                PMR_sub = max(abs(data)) / np.mean(abs(data))


                #**********Max-Inrush Ratio*************************************
                # extreme short range --> can only focus on dataAfter
                # should work. when stacked events happen, their values
                # are part of both parts of the division
                MIR = rms(dataAfter["i"][:sfos])/max(abs(dataAfter["i"][:sfos]))


                #**********Mean-Variance Ratio**********************************
                # Indicator of the current steadiness. To distinguish e.g. Lightbulbs
                # from pure linear loads (e.g. heater)
                # should work, explanation --> Max-Inrush Ratio, Peak-Mean-Ratio
                MVR = np.mean(abs(dataAfter["i"])) / np.var(abs(dataAfter["i"]))

                # better to multiply by 1000?


                #**********Current Over Time************************************
                # calculate the start index for dataBefore, so that the array must not be mirrored
                # better to use the direct substraction method
                length = int(len(dataBefore["i"])/sfos) * sfos
                start = len(dataBefore) - length

                periodsAfter = [dataAfter[i*sfos:(i+1)*sfos] for i in range(int(len(dataAfter["i"])/sfos))]
                periodsBefore = [dataBefore[start+i*sfos: start+(i+1)*sfos] for i in range(int(len(dataBefore["i"])/sfos))]

                if len(periodsAfter) < 25 or len(periodsBefore) < 25:
                    print("Event of {} got skipped, because the array length was too short".format(appliance))
                    continue
                COT_After = [rms(periodsAfter[i]["i"]) for i in range(25)]
                COT_Before = np.mean([rms(periodsBefore[i]["i"]) for i in range(25)])

                COT_Diff = np.array(COT_After[1:]) - COT_Before

                # using the direct subtraction of dataBefore and dataAfter
                if len(periods) < 25:
                    print("Event of {} got skipped, because the array length was too short".format(appliance))
                    continue
                COT = np.array([rms(periods[i]) for i in range(25)])


                #**********Waveform Distortion**********************************
                if len(cleanPeriods) < 10:
                    print("Event of {} got skipped, because the array length was too short".format(appliance))
                    continue
                I_WF = np.mean([cleanPeriods[i] for i in range(10)], axis=0)
                I_WF_N = I_WF/rms(I_WF)
                X = np.arange(sr/lineFreq)
                Y_sin = np.sqrt(2)*np.sin(2 * np.pi * lineFreq * X / sr)
                WFD = sum(abs(Y_sin) - abs(I_WF_N))


                #**********Waveform Approximation*******************************
                WFA = np.float32(np.interp(np.arange(0, len(I_WF), len(I_WF)/20), np.arange(0, len(I_WF)), I_WF))


                #**********Periode to Steady State******************************
                L_thres = 1.0/8.0*(np.max(COT) - np.median(COT)) + np.median(COT)
                try:
                    if len(COT) < 25:
                        print("Event of {} got skipped, because the array length was too short".format(appliance))
                        continue
                    PSS = next(i for i in range(25) if COT[i] < L_thres)
                except: PSS = - 1

                #**********Phase Angle (between voltage and current)************
                COS_PHI = pDiff/sDiff


                """
                #**********VI-Curve*********************************************
                #########using the difference method############################
                # calculate the start index for dataBefore, so that the array must not be mirrored
                #length = int(len(dataBefore["i"])/sfos) * sfos
                #start = len(dataBefore) - length

                periodsAfter = [dataAfter[i*sfos:(i+1)*sfos] for i in range(int(len(dataAfter["i"])/sfos))]
                periodsBefore = [dataBefore[start+i*sfos: start+(i+1)*sfos] for i in range(int(len(dataBefore["i"])/sfos))]
                cleanPeriodsAfter = periodsAfter[1:] # first period has startup peaks
                cleanPeriodsBefore = periodsBefore[:-1]

                U_WF_After = np.mean([cleanPeriodsAfter[i]["v"] for i in range(min(len(cleanPeriodsAfter["v"]), 10))], axis=0)
                I_WF_After = np.mean([cleanPeriodsAfter[i]["i"] for i in range(min(len(cleanPeriodsAfter["i"]), 10))], axis=0)
                U_WF_Before = np.mean([cleanPeriodsBefore[i]["v"] for i in range(min(len(cleanPeriodsBefore["v"]), 10))], axis=0)
                I_WF_Before = np.mean([cleanPeriodsBefore[i]["i"] for i in range(min(len(cleanPeriodsBefore["i"]), 10))], axis=0)
                U_WF_Diff = U_WF_After - U_WF_Before
                I_WF_Diff = I_WF_After - I_WF_Before
                U_Norm_Diff = U_WF_Diff/max(-1*min(U_WF_Diff), max(U_WF_Diff))
                I_Norm_Diff = I_WF_Diff/max(-1*min(I_WF_Diff), max(I_WF_Diff))
                """

                #**********VI-Curve*********************************************
                dataI = data
                dataV = np.array(dataAfter[:e]["v"] - dataBefore[-e:]["v"])
                periodsI = [dataI[i*sfos:(i+1)*sfos] for i in range(int(len(data)/sfos))]
                periodsV = [dataV[i*sfos:(i+1)*sfos] for i in range(int(len(data)/sfos))]
                cleanPeriodsI = periodsI[1:]
                cleanPeriodsV = periodsV[1:]

                # Calculate the VI-Trajectory
                if len(cleanPeriodsI) < 10 or len(cleanPeriodsV) < 10:
                    print("Event of {} got skipped, because the array length was too short".format(appliance))
                    continue
                I_WF = np.mean([cleanPeriodsI[i] for i in range(10)], axis=0)
                U_WF = np.mean([cleanPeriodsV[i] for i in range(10)], axis=0)
                # normalize to compensate voltage fluctuations
                U_NORM = U_WF/max(-1*min(U_WF), max(U_WF))
                I_NORM = I_WF/max(-1*min(I_WF), max(I_WF))
                # see if samples for one sine is integer multiple of 20
                # if not upsample it to next
                if sfos/20.0 - int(sfos/20.0) > 0:
                    n = int((int(sfos/20.0) + 1) * 20)
                    U_NORM2 = scipy.signal.resample(U_NORM, n)
                    I_NORM2 = scipy.signal.resample(I_NORM, n)
                    VIT = list(U_NORM2[::int(len(U_NORM2)/20)]) + list(I_NORM2[::int(len(I_NORM2)/20)])
                else:
                    VIT = list(U_NORM[::int(sfos/20)][:20]) + list(I_NORM[::int(sfos/20)][:20])


                #**********Inrush Current Ratio*********************************
                # compare the current data from start to end
                # change: start from 1 instead of 0
                ICR = rms(periods[1])/rms(periods[-1])


                #*********************FREQUENCY DOMAIN**************************
                # use the welch method to get frequency spectrum
                # Welch's method for estimating power spectra is carried out by
                # dividing the time signal into successive blocks, forming
                # the periodogram for each block, and averaging.

                NFFT = len(data)
                FFT, f = fft(data, sr, win="hamming")

                #**********Harmonic Energy Distribution*************************
                # Vector of 20 values; Magnitudes of 50, 100, 150, ..., 1000Hz
                # Calculate Harmonic Energy Distribution
                HEDFreqs = [float(i*lineFreq) for i in range(1, 20)]

                # Use margin around goal freq to calculate single harmonic magnitude
                indices = [[i for i, f_ in enumerate(f) if abs(f_-freq_) < 3] for freq_ in HEDFreqs]
                Harm = [sum(FFT[ind]) if len(ind) > 0 else 0 for ind in indices]
                if Harm[0] == 0:
                    #raise AssertionError("Could not determine Harmonics, maybe increase FFT size")
                    print("Could not dertermine Harmonics")
                    continue
                HED = Harm[1:]/Harm[0]

                #**********Total Harmonic Distortion****************************
                THD = 10*np.log10(sum(Harm[1:6])/Harm[0])

                #**********Spectral Flatness************************************
                # Measure for energy distribution in freq spectrum.
                # A Value of 1.0 is equivalent to white noise. The closer to 0, the
                # stronger are individual frequencies (linear loads -> low values)
                NormFFT = FFT/max(FFT)
                SPF = (scipy.stats.mstats.gmean(NormFFT))/(sum(NormFFT)/len(NormFFT))

                #**********Odd-Even Harmonics Ratio*****************************
                # Some (most) appliances show an imbalanced odd (150, 250, 350Hz ...)
                # to even (100, 200, 300Hz, ...) rate
                OER = np.mean(HEDFreqs[1::2])/np.mean(HEDFreqs[2::2])

                #**********Tristimulus******************************************
                # Energy of different harmonic groups (low, medium, high - harmonics)
                TRI = [Harm[1]/sum(Harm), sum(Harm[2:5])/sum(Harm), sum(Harm[5:11])/sum(Harm)]

                #**********Spectral Centroid************************************
                # Same as temporal centroid but for the frequency spectrum
                SC = sum([xf*(bin*(sr/NFFT)) for bin, xf in enumerate(FFT)]) / sum(FFT)




                # sub means that the values are calculated by directly
                # subtracting the current before and after event instead of subtracting
                # the calculated values before and after the event
                features = {
                    "p": pDiff,
                    "q": qDiff,
                    "s": sDiff,
                    "R": RDiff, #ok
                    "Y": YDiff, #ok
                    "cfDiff": cfDiff,
                    "CF": cf_sub, #(sub)
                    #"FFDiff": FFDiff, #schlecht
                    "FF": FF_sub, #(sub)
                    "LAT": LAT, #sehr gut (sub)
                    "TC": TC, #sehr gut (sub)
                    "PNR": PNR, #gut (sub)
                    "MAMI": MAMI, #gut (sub)
                    "MAMI_Diff": MAMI_Diff, #gut
                    "PMR": PMR, #sehr gut
                    #"PMR_sub": PMR_sub, #schlecht
                    "MIR": MIR, #sehr gut
                    "MVR": MVR,
                    "WFD": WFD,
                    #"WFA": WFA,
                    #"COT_Diff": COT_Diff, #schlecht
                    #"COT": COT, #ok (sub)
                    "PSS": PSS, #gut
                    "COS_PHI": COS_PHI, #gut
                    "ICR": ICR, #ok
                    "HED": HED,
                    "THD": THD,
                    "SPF": SPF,
                    "OER": OER,
                    "TRI": TRI,
                    "SC": SC,


                    "timestamp": datetime.fromtimestamp(timestamp),
                    "phase": phase,
                }


                if array == True:
                    features["U_Norm"] = U_NORM #better
                    features["I_Norm"] = I_NORM #better
                    features["COT"] = COT
                    features["WFA"] = WFA
                    features["VIT"] = VIT

                eventFeatures[phase][direction].append(features)
                eventAppliance[phase][direction].append(appliance)

                """
                if appliance not in applianceFeatures.keys():
                    applianceFeatures[appliance] = {}
                if direction not in applianceFeatures[appliance].keys():
                    applianceFeatures[appliance][direction] = []
                applianceFeatures[appliance][direction].append(features)
                """

                del x_data

    return eventFeatures, eventAppliance


# no longer used for the current classification approach
"""
def calculateMean(features):
    featuresMean = {}
    for feature in features[0].keys():
        if feature == "timestamp" or feature == "phase":
            continue
        if feature == "U_Norm" or feature == "I_Norm" or feature == "COT" or feature == "COT_Diff" or feature == "WFA" or feature == "VIT":
            sum = [features[0][feature]]
            length = len(features)
            for i in range(1, length):
                sum = np.append(sum, [features[i][feature]], axis=0)
            mean = np.mean(sum, axis=0)
        else:
            sum = 0
            for i in range(0, len(features)):
                sum += features[i][feature]
            mean = sum / len(features)
        featuresMean[feature] = mean
    return featuresMean
"""

# in the NULL-class many different undetected appliances are grouped together.
# to help the classifier this function will group similar events of the NUll-class
# in subgroups

#which features should be checked for similarities
CHECK_FEATURES = {"p", "q", "s"}

def groupNullClass(eventFeatures, eventAppliance):
    nullFeatures = {}
    featuresSaved = {}
    for phase in ["l1", "l2", "l3"]:
        nullFeatures[phase] = {}
        featuresSaved[phase] = {}
        for direction in ["up", "down"]:
            nullFeatures[phase][direction] = []
            featuresSaved[phase][direction] = []
            for i in range(len(eventAppliance[phase][direction]) - 1, -1, -1):
                if eventAppliance[phase][direction][i] == "NULL":
                    features_sel = [eventFeatures[phase][direction][i][f] for f in CHECK_FEATURES]
                    featuresSaved[phase][direction].append(eventFeatures[phase][direction][i])
                    nullFeatures[phase][direction].append(features_sel)
                    del eventAppliance[phase][direction][i]
                    del eventFeatures[phase][direction][i]

    # group NULL-events
    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]:
            clustering = DBSCAN(eps=5, min_samples=5).fit(nullFeatures[phase][direction])
            labels = clustering.labels_
            # add the clustered events to the other events
            print(phase, direction)
            print(set(labels))
            if DELETE_EVENTS_WITHOUT_CLUSTER == True:
                for i in range(len(labels) - 1, -1, -1):
                    if labels[i] == -1:
                        labels = np.delete(labels, i)
                        del featuresSaved[phase][direction][i]
                eventFeatures[phase][direction].extend(featuresSaved[phase][direction])
            else:
                eventFeatures[phase][direction].extend(featuresSaved[phase][direction])
            for label in labels:
                if label == -1:
                    eventAppliance[phase][direction].append("NULL")
                else:
                    eventAppliance[phase][direction].append("NULL_{0}".format(label))

    return eventAppliance, eventFeatures


def saveDataPickle(x, y):
    #filename = "FIREDEventsExtracted_" + datetime.fromtimestamp(recordingRange[0]).strftime("%d_%m_%Y") + "-" + datetime.fromtimestamp(recordingRange[1]).strftime("%d_%m_%Y") + ".pickle"
    filename = "eventFeatures_difference_final_nodbscan_21_06_12_07.pickle"
    filehandler = open(os.path.join("data", filename),"wb")
    pickle.dump([x, y], filehandler)
    filehandler.close()
    return


print("Loading pickle-file with the extracted events")
x = loadDataPickle()
print("Extracting features from all given events...")
features, appliances = extractFeatures(x, array=True)

if CLUSTER_NULL_EVENTS == True:
    print("Grouping similar events of the NULL-class into subclasses")
    appliances, features = groupNullClass(features, appliances)

"""
print("Grouping different events of the same appliance and calculate the mean of their features")
featuresMean = groupSubEvents(features)
pp(featuresMean)
"""

print("Saving features in pickle file")
saveDataPickle(features, appliances)
print("Finished successfully")
