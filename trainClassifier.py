# for printing dicts in a more readable way
from beeprint import pp

# various modules
import pickle
import numpy as np
import os

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
N_JOBS = -1 # -1 means using all processors
# should the classifiers be tested or should the whole set be used for training
TEST = True
PRINT_CONFUSION_MATRIX = True

selectedFeaturesCleaning = ["COT",
                            "MVR",
                            "PMR",
                            "s",
                            "FF",
                            "COS_PHI",
                            "p"]


selectedFeaturesDifference = ['HED',
                              'Y',
                              'MVR',
                              's',
                              'q',
                              'SC',
                              'THD',
                              'WFD']

if METHOD == "cleaning":
    selectedFeatures = selectedFeaturesCleaning
elif METHOD == "difference":
    selectedFeatures = selectedFeaturesDifference

selectedModels = ["knn",
                  "rf ",
                  "svm",
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


def modelSelection(m, x_train, x_test, y_train, y_test, y, phase, direction):
    #perform a grid search
    gridsearch_cv = GridSearchCV(m["classifier"], m["grid"], cv=CROSS_VALIDATION_SPLITS, scoring="f1_" + F1_AVERAGE, n_jobs = N_JOBS)
    gridsearch_cv.fit(x_train, y_train)

    results_dict = {}
    results_dict["classifier"] = gridsearch_cv.best_estimator_
    results_dict["best_params"] = gridsearch_cv.best_params_
    #print(results_dict["best_params"])
    results_dict["F1_CV"] = gridsearch_cv.best_score_

    if TEST == True:
        y_pred = gridsearch_cv.best_estimator_.predict(x_test)
        results_dict['F1'] = f1_score(y_test, y_pred, average=F1_AVERAGE)
        results_dict['Precision'] = precision_score(y_test, y_pred, average=F1_AVERAGE)
        results_dict['Recall'] = recall_score(y_test, y_pred, average=F1_AVERAGE)
        results_dict['Accuracy'] = accuracy_score(y_test, y_pred)
        #print(results_dict['F1'])

        if PRINT_CONFUSION_MATRIX == True:
            confusionMatrix(results_dict, x_test, y_test, y, phase, direction, m)
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


def classify(features, appliances):
    print("Generating the test and train set for classification \n")
    combindedResults = {"l1": {"up": {}, "down": {}}, "l2": {"up": {}, "down": {}}, "l3": {"up": {}, "down": {}}}
    for phase in ["l1", "l2", "l3"]: #using keys so that only existing keys will be used
        for direction in appliances[phase].keys():
            print("Working on {0} with direction {1}".format(phase, direction))
            # extract only the features specified in the selectedFeatures
            features_sel = [[x[f] for f in selectedFeatures] for x in features[phase][direction]]
            if TEST == True:
                # using only numbers can not be realized, because we use devices other than powermeters. (e.g. "espresso machine" has no number)
                f_train, f_test, a_train, a_test = splitData(features_sel, appliances[phase][direction])
                f_train = flatten(f_train)
                f_test = flatten(f_test)
            else:
                # no data split is necessary because the classifier won't be tested
                f_train = features_sel
                a_train = appliances[phase][direction]
                f_test = 0
                a_test = 0
                f_test_scaled = 0
                f_train = flatten(f_train)

            # scaling the training-set
            scaler = preprocessing.StandardScaler().fit(f_train)
            f_train_scaled = scaler.transform(f_train)
            if TEST == True: f_test_scaled = scaler.transform(f_test)

            # delete the data that is no longer used
            del f_train
            del f_test

            # start the actual classification process
            for m in models:
                if m["name"] not in selectedModels: continue
                print("{}...".format(m["name"]))
                results_dict = modelSelection(m, f_train_scaled, f_test_scaled, a_train, a_test, appliances[phase][direction], phase, direction)

                combindedResults[phase][direction][m["name"]] = results_dict

    return combindedResults, scaler


def printClassificationResulst(results):
    formatter = "{:>12}{:>10}{:>9}{:>14}"
    formatter2 =  "{:>0}{:>9}{:>7}{:>9}{:>24}"
    formatter3 =  "{:>0}{:>7}{:>7}{:>9}{:>24}"
    print(formatter.format("F1", "F1_CV", "Name", "Parameter"))
    for phase in ["l1", "l2", "l3"]:
        for direction in ["up", "down"]:
            for m in results[phase][direction].keys():
                k = results[phase][direction][m]
                if direction == "up":
                    print(formatter2.format(phase + "_" + direction, round(k["F1"], 2), round(k["F1_CV"], 2), m, str(k["best_params"])))
                if direction == "down":
                    print(formatter3.format(phase + "_" + direction, round(k["F1"], 2), round(k["F1_CV"], 2), m, str(k["best_params"])))
    return


def saveDataPickle(x, y, z):
    #filename = "FIREDEventsExtracted_" + datetime.fromtimestamp(recordingRange[0]).strftime("%d_%m_%Y") + "-" + datetime.fromtimestamp(recordingRange[1]).strftime("%d_%m_%Y") + ".pickle"
    filename = "trainedClassifiers.pickle"
    filehandler = open(os.path.join("data", filename),"wb")
    pickle.dump([x, y, z], filehandler)
    filehandler.close()
    return


def plotConfusionMatrix(cm, target_names, title='Confusion matrix', show=True, cmap=None, normalize=True, f1=None, dpi=100, figsize=(20, 20)):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None: cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize, dpi=dpi)

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    aspect = 20
    pad_fraction = 0.5

    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.colorbar(im, cax=cax)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)


    thresh = cm.max() / 10 if normalize else cm.max() / 2
    # Leve out numbers for larger matrices
    if len(target_names) < 50:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                num = str(round(cm[i, j],2))
                while num[-1] == '0': num = num[:-1]
                if num[-1] == '.': num = num[:-1]
                ax.text(j, i, "{}".format(num),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    else:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)

    plt.tight_layout()
    ax.set_ylabel('True label')
    string = 'Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass)
    if f1 is not None: string += "; F1={:0.4f}".format(f1)
    ax.set_xlabel(string)
    if show:
        plt.show(block=show)
    return plt.gcf()


def confusionMatrix(results, x_test, y_test, y, phase, direction, m):
    classifier = results["classifier"]
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    y_set = sorted(list(set(y)))
    fig = plotConfusionMatrix(cm, y_set, show=False, figsize=(7,7), f1=results["F1"])
    fig.delaxes(fig.axes[1])
    fig.gca().set_title(m["name"] + " applied to \"" + phase + "\" with direction \"" + direction + "\n")
    fig.gca().set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.show()


print("Loading the features of all events...")
features, appliances = loadDataPickle(FILENAME_CATEGORIZED_EVENTS)


print("Checking if there are enough samples for all appliances")
features_clean, appliances_clean = checkSampleSize(features, appliances)


print("\nStarting the classification process...")
results, scaler = classify(features_clean, appliances_clean)


if TEST == False:
    print("\nSaving the trained classifiers for all phases and directions\n")
    saveDataPickle(results, selectedFeatures, scaler)

else:
    print("\nClassification process completed. Showing Classification results...\n")
    printClassificationResulst(results)

    if PRINT_CONFUSION_MATRIX == True:
        print("\nPlotting the confusion matrix")
        plt.show(block=True)
