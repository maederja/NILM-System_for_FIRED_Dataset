# for printing dicts in a more readable way
from beeprint import pp

# various modules
import pickle
import numpy as np
import os
import json

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
# import classification algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# scoring the results
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Plotting
import matplotlib.pyplot as plt

#median
import statistics

# Warnings
import warnings
warnings.filterwarnings("ignore", category = UserWarning)


#%%%%%%%%%SETUP%%%%%%%%%%
# choose which method for extracting the features was used
# choose between "cleaning" and "difference"
METHOD = "difference"
#FILENAME_ALL_EVENTS = "allEventFeaturesClassification_" + METHOD + ".pickle"
FILENAME_CATEGORIZED_EVENTS = "eventFeatures_" + METHOD + "_final_nodbscan_21_06_12_07.pickle"

#set variables for classification
MIN_SAMPLES = 3
RANDOM_STATE = 12
F1_AVERAGE = "macro"
CROSS_VALIDATION_SPLITS = 5
N_JOBS = 4 # -1 means using all processors
# should the classifiers be tested or should the whole set be used for training

selectedFeatures = ["p",
                    "q",
                    "s",
                    "R",
                    "Y",
                    "CF",
                    "FF",
                    "LAT",
                    "TC",
                    "PNR",
                    "MAMI",
                    "PMR",
                    "MIR",
                    "MVR",
                    "WFD",
                    "WFA",
                    "COT",
                    "PSS",
                    "COS_PHI",
                    "VIT",
                    "ICR",
                    "HED",
                    "THD",
                    "SPF",
                    "OER",
                    "TRI",
                    "SC",
                    #"U_NORM",
                    #"I_NORM",
                    ]

# specify the models which should be used for classification
models = [{'name': 'knn', 'label':'K Nearest Neighbors',
           'classifier':KNeighborsClassifier(),
           'grid': {"n_neighbors":np.arange(20)+1},
          },

          {'name': 'rf ', 'label': 'Random Forest',
           'classifier': RandomForestClassifier(random_state=RANDOM_STATE),
           'grid': {
                    'criterion' :['gini'],
                    'n_estimators' : [10,50,100,1000],
                    'max_features': ['auto'],
                    'max_depth' : [10*i for i in range(1,11)],
                    }
           },

           {'name': 'svm', 'label': 'SVC (RBF)',
            'classifier':SVC(random_state=RANDOM_STATE),
            'grid': {
                     'kernel': ['rbf'],
                     'C': [0.01, 0.1, 1, 100, 1000],
                     'gamma': [10000, 1000, 100, 10, 1, 0.1, 0.01],
                     },
           },
]

selectedModels = ["knn",
                  "rf ",
                  "svm",
                  ]


def loadDataPickle(filename):
    filehandler = open(os.path.join("data", filename),"rb")
    file = pickle.load(filehandler)
    filehandler.close()
    x = file[0]
    y = file[1]

    return x, y


def checkSampleSize(featuresAll, appliancesAll):
    features_clean = {"l1": {}, "l2": {}, "l3": {}}
    appliances_clean = {"l1": {}, "l2": {}, "l3": {}}
    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]:
            features = featuresAll[phase][direction]
            appliances = appliancesAll[phase][direction]

            appliances_set = set(appliances)
            for element in appliances_set:
                indices = [i for i, j in enumerate(appliances) if j == element]
                if len(indices) < MIN_SAMPLES:
                    print(element + " deleted because there are too few samples")
                    for i in reversed(indices):
                        del features[i]
                        del appliances[i]

            if len(appliances) != 0:
                features_clean[phase][direction] = features
                appliances_clean[phase][direction] = appliances

    return features_clean, appliances_clean



def splitData(x, y):
    # test_size is the size of the array which shouls be used for testing (0...0.2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    return x_train, x_test, y_train, y_test


def classify(features, appliances, selectFeatures):
    combindedResults = {"l1": {"up": {}, "down": {}}, "l2": {"up": {}, "down": {}}, "l3": {"up": {}, "down": {}}}
    for phase in ["l1", "l2", "l3"]:
        for direction in appliances[phase].keys():
            # extract only the features specified in the selectedFeatures
            features_sel = [[x[f] for f in selectFeatures] for x in features[phase][direction]]
            # using only numbers can not be realized, because we use devices other than powermeters. (e.g. "espresso machine" has no number)
            f_train, f_test, a_train, a_test = splitData(features_sel, appliances[phase][direction])

            f_train = flatten(f_train)
            f_test = flatten(f_test)

            # scaling the training-set
            scaler = preprocessing.StandardScaler().fit(f_train)
            f_train_scaled = scaler.transform(f_train)
            f_test_scaled = scaler.transform(f_test)

            # delete the data that is no longer used
            del f_train
            del f_test

            # start the actual classification process
            for m in models:
                if m["name"] != selectedModel: continue
                #print("{}...".format(m["name"]))
                results_dict = modelSelection(m, f_train_scaled, f_test_scaled, a_train, a_test, appliances[phase][direction], phase, direction)

                combindedResults[phase][direction][m["name"]] = results_dict

    return combindedResults


def modelSelection(m, x_train, x_test, y_train, y_test, y, phase, direction):
    #perform a grid search
    gridsearch_cv = GridSearchCV(m["classifier"], m["grid"], cv=CROSS_VALIDATION_SPLITS, scoring="f1_" + F1_AVERAGE, n_jobs=N_JOBS)
    gridsearch_cv.fit(x_train, y_train)

    results_dict = {}
    #results_dict["classifier"] = gridsearch_cv.best_estimator_
    #results_dict["best_params"] = gridsearch_cv.best_params_

    results_dict["F1_CV"] = gridsearch_cv.best_score_

    y_pred = gridsearch_cv.best_estimator_.predict(x_test)
    results_dict['F1'] = f1_score(y_test, y_pred, average=F1_AVERAGE)

    return results_dict


# scaler can't scale a list within a list. So flatten the lists into a single list
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


def saveDataJSON(x, y, z, selectedModel):
    #filename = "FIREDEventsExtracted_" + datetime.fromtimestamp(recordingRange[0]).strftime("%d_%m_%Y") + "-" + datetime.fromtimestamp(recordingRange[1]).strftime("%d_%m_%Y") + ".pickle"
    filename = "TestFeatures_{0}_difference_orig_noNull_3weeks_median_mean.json".format(selectedModel)
    filehandler = open(os.path.join("results/features_test", filename),"w")
    json.dump([x, y, z], filehandler)
    filehandler.close()
    return


print("Loading the features of all events...")
features, appliances = loadDataPickle(FILENAME_CATEGORIZED_EVENTS)

print("Checking if there are enough samples for all appliances")
features_clean, appliances_clean = checkSampleSize(features, appliances)

print("\nStarting the process of determining the best feature combination")

for selectedModel in selectedModels:
    print("\nWorking with {0}-classifier:".format(selectedModel))
    bestCombinations = []
    scores = []
    scoresDict = []

    temp_bestCombinations = []
    temp_score = 0
    temp_scoresDict = []
    better = False

    i = 0
    for feature in selectedFeatures:
        print("Iteration: {0} with feature: {1}". format(i, feature))
        selectFeatures = [feature]
        results = classify(features, appliances, selectFeatures)
        elements = []
        for phase in ["l1", "l2", "l3"]:
            for direction in ["up", "down"]:
                for m in models:
                    if m["name"] != selectedModel: continue
                    elements.append(results[phase][direction][m["name"]]["F1_CV"])

        score = statistics.median(elements)
        #score = np.mean(elements)
        if score > temp_score:
            temp_score = score
            temp_bestCombinations = [feature]
            temp_scoresDict = results
            better = True


    bestCombinations = [temp_bestCombinations]
    scores = [temp_score]
    scoresDict = [temp_scoresDict]

    print(bestCombinations, scores)

    while better == True:
        better = False
        i += 1
        temp_score = 0
        for feature in selectedFeatures:
            # check if this feature has already been used
            if feature in bestCombinations[len(bestCombinations) - 1]: continue
            selectFeatures = bestCombinations[len(bestCombinations) - 1].copy()
            selectFeatures.append(feature)
            print("Iteration: {0} with features: {1}". format(i, selectFeatures))

            results = classify(features, appliances, selectFeatures)
            elements = []
            for phase in ["l1", "l2", "l3"]:
                for direction in ["up", "down"]:
                    for m in models:
                        if m["name"] != selectedModel: continue
                        elements.append(results[phase][direction][m["name"]]["F1"])

            #score = statistics.median(elements)
            score = np.mean(elements)
            if score > temp_score and score > scores[len(scores) - 1]:
                temp_score = score
                temp_bestCombinations = selectFeatures
                temp_scoresDict = results
                better = True

        if better == True:
            bestCombinations.append(temp_bestCombinations)
            scores.append(temp_score)
            scoresDict.append(temp_scoresDict)
            print(bestCombinations, scores)

    saveDataJSON(bestCombinations, scores, scoresDict, selectedModel)
