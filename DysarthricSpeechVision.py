import utilities
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout , SpatialDropout2D
from tensorflow.keras.callbacks import History
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
import optuna


SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))

model_name=input("Which model do you want to train/test? ")
train_set_path = SETTINGS_DIR+'/images/Train/'
test_set_path = SETTINGS_DIR+"/images/Test/"
dnn_file_name_structure = SETTINGS_DIR + "/Models/cnn_" + model_name + ".json"
training_dynamics_path = SETTINGS_DIR+'/Training Performance/TrainingDynamics'+model_name+'.csv'
dnn_file_name_weights = SETTINGS_DIR + "/Models/cnn_weight_"+model_name+".h5"

batch_size = 64
image_input_size = (150, 150)
classes = utilities.get_no_folders_in_path(test_set_path)
print("Number of Classes:", classes)


def model_compile(model):

    opt = optimizers.Adam(lr=0.002)
    
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])


def model_compile_adam(model,lr_value):
    opt = optimizers.Adam(lr=lr_value)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])


def model_compile_rms(model,lr_value):
    opt = optimizers.RMSprop(lr=lr_value)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])


def model_compile_sgd(model,lr_value):
    opt = optimizers.SGD(lr=lr_value)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

def model_compile_optuna(model, optimizer):

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])


def get_model():

    droprate=0.5

    classifier = Sequential()

    classifier.add( Convolution2D(  filters=32, kernel_size=(3,3), 
                                  input_shape= (*image_input_size,3), 
                                  activation='relu')  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=32, kernel_size=(3,3),activation='relu'  )  )
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(droprate))


    classifier.add( Convolution2D(  filters=64, kernel_size=(3,3),activation='relu'  )  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=64, kernel_size=(3,3),activation='relu'  )  )
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(droprate))

    classifier.add( Convolution2D(  filters=128, kernel_size=(3,3),activation='relu'  )  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=128, kernel_size=(3,3),activation='relu'  )  )
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(droprate))

    classifier.add( Convolution2D(  filters=256, kernel_size=(3,3),activation='relu'  )  )
    classifier.add( SpatialDropout2D(droprate) )
    classifier.add( Convolution2D(  filters=256, kernel_size=(3,3),activation='relu'  )  )
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D (pool_size=(2,2) ) )
    #classifier.add(Dropout(0.5))
    classifier.add(Dropout(droprate))

    classifier.add (Flatten( ) )

    classifier.add(Dense (units=classes, activation='softmax' ))
    classifier.summary()

    return classifier


def read_epoch():
    if os.path.exists(training_dynamics_path):
        
        # First check the csv file has headres and add then if missing
        try:
            training_dynamics=pd.read_csv(training_dynamics_path)
            training_dynamics["Epoch"][len(training_dynamics)-1]
        except:
            df = pd.read_csv(training_dynamics_path, header=None, index_col=None)
            df.columns = columns=["","Epoch","TrainingLoss", "TrainingAccuracy","ValidationLoss","ValidationAccuracy"]
            df.to_csv(training_dynamics_path, index=False)
        training_dynamics=pd.read_csv(training_dynamics_path)               
        return training_dynamics["Epoch"][len(training_dynamics)-1]
        
    else:
        return 0

def load_model():
    # Loading the CNN
    json_file = open(dnn_file_name_structure, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(dnn_file_name_weights)
    model_compile(model)
    return model

def save_model(model, is_max_val_inclluded=False,max_val=None, ep=None):
    # Save/overwrite the model
    if (is_max_val_inclluded):
        json_file_name = SETTINGS_DIR+"/Models/cnn_"+model_name+"_MAX_VALUE"+"_"+str(max_val)+"_"+str(ep)+".json"
        wights_file_name = SETTINGS_DIR+"/Models/cnn_weight_"+model_name+"MAX_VALUE"+"_"+str(max_val)+"_"+str(ep)+".h5"
        '''
        # Delete previously stored models for this speaker
        for directory, s, files in os.walk(SETTINGS_DIR+"/Models/"):
            for f in files:
                if speaker_name in f:
                    file_path=directory+"/"+f
                    os.remove(file_path)
        '''
    else:
        json_file_name = dnn_file_name_structure
        wights_file_name = dnn_file_name_weights
    
    model_json = model.to_json()
    with open(json_file_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(wights_file_name)
    
def save_training_dynamics(epoch,history,with_header=False):
    training_dynamics=pd.DataFrame(
        data = [ [epoch, history.history['loss'][-1] ,  history.history['acc'][-1],
                history.history['val_loss'][-1],  history.history['val_acc'][-1] ]],
        columns=["Epoch","TrainingLoss", "TrainingAccuracy","ValidationLoss","ValidationAccuracy"]
    )
    if (with_header):
        with open(training_dynamics_path, 'a') as csv_file:
            training_dynamics.to_csv(csv_file, header=True)
    else:
        with open(training_dynamics_path, 'a') as csv_file:
            training_dynamics.to_csv(csv_file, header=False)
            
def visualize_training():
    import matplotlib.pyplot as plt
    if (os.path.isfile(training_dynamics_path) == False ):
        print ("Training dynamics file is not found.")
        return
    try:
        training_dynamics=pd.read_csv(training_dynamics_path)
        loss_values = training_dynamics["TrainingLoss"]
        val_loss_values = training_dynamics["ValidationLoss"]
        epochs = range(1, len (training_dynamics['Epoch'])+1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # Ploting Accuracy
        loss_values = training_dynamics["TrainingAccuracy"]
        val_loss_values = training_dynamics["ValidationAccuracy"]
        epochs = range(1, len (training_dynamics['Epoch'])+1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
     
    except:
        df = pd.read_csv(training_dynamics_path, header=None, index_col=None)
        df.columns = ["","Epoch","TrainingLoss", "TrainingAccuracy","ValidationLoss","ValidationAccuracy"]
        df.to_csv(training_dynamics_path, index=False)
        visualize_training()
    
def get_train_test_sets():
        from keras.preprocessing.image import ImageDataGenerator
        
        # https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
        train_datagen = ImageDataGenerator(
                    rescale=1./255,
            width_shift_range=0.30,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            horizontal_flip=False)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # If shuffle=False then the validation results will be different from classifier.predict_generator()
        print ("Setting training date...")
        training_set = train_datagen.flow_from_directory(
            train_set_path,
            target_size=image_input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)
        
        print ("Setting testing date...")
        test_set = test_datagen.flow_from_directory(
           test_set_path,
            target_size=image_input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

        return training_set, test_set


def get_Predictions_and_Labels():
    model = load_model()

    predictions = model.predict_generator(test_set)

    true_label = test_set.classes

    return predictions, true_label

def manual_testing_FRR_FAR(predictions, true_labels, threshold):

    num_rejections = 0
    num_acceptances = 0
    total_valid_samples = 0
    total_invalid_samples = 0

    for pred, true_label in zip(predictions, true_labels):
        confidence_score = pred.max()
        predicted_class = pred.argmax()

        if true_label == predicted_class:
            total_valid_samples += 1
            if confidence_score < threshold:
                num_rejections += 1

        else:
            total_invalid_samples += 1
            if confidence_score >= threshold:
                num_acceptances += 1

    FRR = num_rejections / total_valid_samples if total_valid_samples > 0 else 0
    FAR = num_acceptances / total_invalid_samples if total_invalid_samples > 0 else 0

    return FRR, FAR



def plot_ROC_curve(FPR, TPR):

    import matplotlib.pyplot as plt

    normFPR = []
    normTPR = []

    FPRall = np.zeros(len(FPR[0]))
    TPRall = np.zeros(len(TPR[0]))

    for i in range(len(FPR[0])):
        FPRall[i] += (FPR[0][i] + FPR[1][i] + FPR[2][i] + FPR[3][i]) / 4

    for i in range(len(FPR[0])):
        TPRall[i] += (TPR[0][i] + TPR[1][i] + TPR[2][i] + TPR[3][i]) / 4

    for fpr, tpr in zip(FPRall, TPRall):
        normFPR.append((fpr - min(FPRall)) / (max(FPRall) - min(FPRall)))
        normTPR.append((tpr - min(TPRall)) / (max(TPRall) - min(TPRall)))


    plt.plot(normFPR, normTPR, 'r', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def calculate_FPR_TPR(thresholds):
    
    model = load_model()
    
    y_probs = model.predict_generator(test_set)
    
    
    FPRs = np.zeros((4,len(thresholds)))
    TPRs = np.zeros((4,len(thresholds)))

    for class_i in range(4):
        thresh_count = 0
        for threshold in thresholds:

            y_pred = np.argmax(y_probs, axis=1)
            max_probs = np.max(y_probs, axis=1)

            test_set_bin = np.zeros(len(test_set.classes),dtype=int)

            for i in range(len(y_pred)):
                
                if test_set.classes[i] == class_i :
                    test_set_bin[i] = 1
                else:
                    test_set_bin[i] = 0

                if max_probs[i] < threshold:
                    y_pred[i] = 0
                else:
                    if y_pred[i] == class_i:
                        y_pred[i] = 1
                    else:
                        y_pred[i] = 0


            cm = np.zeros((2,2))

            for i in range(len(y_pred)):
                if y_pred[i] == test_set_bin[i] and y_pred[i] == 1:
                    cm[0][0] += 1
                elif y_pred[i] == test_set_bin[i] and y_pred[i] == 0:
                    cm[1][1] += 1
                elif y_pred[i] != test_set_bin[i] and y_pred[i] == 1:
                    cm[0][1] += 1
                elif y_pred[i] != test_set_bin[i] and y_pred[i] == 0:
                    cm[1][0] += 1

            TP = cm[0][0]
            FN = cm[1][0]
            FP = cm[0][1]
            TN = cm[1][1]

            P = TP + FN
            N = FP + TN

            if N != 0:
                FPR = FP / N
            else:
                FPR = 0
            if P != 0:
                TPR = TP / P
            else:
                TPR = 0

            FPRs[class_i][thresh_count] = FPR
            TPRs[class_i][thresh_count] = TPR

            thresh_count += 1

    return FPRs, TPRs


        
def create_Confusion_Matrix():
    import matplotlib.pyplot as plt
    
    model = load_model()
    
    y_pred = model.predict_generator(test_set)

    y_pred = np.argmax(y_pred, axis=1)    

    cm = confusion_matrix(test_set.classes, y_pred)

    cm_df = pd.DataFrame(cm,
                        index = ['High','Low','Medium','Very Low'],
                        columns = ['High','Low','Medium','Very Low'])

    cm_df.index = pd.CategoricalIndex(cm_df.index,categories=["Very Low", "Low", "Medium", "High"])
    cm_df.sort_index(level=0, inplace=True)
    cm_df = cm_df.reindex(columns=["Very Low", "Low", "Medium", "High"])


    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show
    
    return cm


def roc_curve():
    import numpy as np
    import matplotlib.pyplot as plt

    thresholds = np.arange(0.0, 1.0, 0.01)
    FPR, TPR = calculate_FPR_TPR(thresholds=thresholds)


    normFPR = []
    normTPR = []

    FPRall = np.zeros(len(FPR[0]))
    TPRall = np.zeros(len(TPR[0]))

    for i in range(len(FPR[0])):
        FPRall[i] += (FPR[0][i] + FPR[1][i] + FPR[2][i] + FPR[3][i]) / 4

    for i in range(len(FPR[0])):
        TPRall[i] += (TPR[0][i] + TPR[1][i] + TPR[2][i] + TPR[3][i]) / 4

    TPRall = 1 - TPRall

    for fpr, tpr in zip(FPRall, TPRall):
        normFPR.append((fpr - min(FPRall)) / (max(FPRall) - min(FPRall)))
        normTPR.append((tpr - min(TPRall)) / (max(TPRall) - min(TPRall)))


    index = 0

    #calculate eer
    from math import isclose
    for i, (fpr, tpr) in enumerate(zip(normFPR, normTPR)):
        print(fpr,' ', tpr)
        if isclose(fpr, tpr, abs_tol=0.01):
            print('i', i, '-', fpr, '=', tpr)
            index = i

    x_point = normTPR[index]
    y_point = normFPR[index]

    print('EER: ', x_point)

    # calculate mindcf
    c_miss = 1
    c_fa = 1
    p_target = 0.05

    dcf_values = []
    min_dcf_index = 0

    for i, (fpr, fnr) in enumerate(zip(normFPR, normTPR)):
        dcf_values.append(c_miss * p_target * fpr + c_fa * (1 - p_target) * fnr)
        if fpr < dcf_values[i] < (fpr + p_target):
            min_dcf_index = i

    print("index", min_dcf_index)

    print("minDCF: ", 1 - dcf_values[min_dcf_index])

    plt.plot(normTPR, normFPR, 'r', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.scatter(x_point, y_point, color='red', label=f"EER {x_point}")
    plt.annotate("EER", (normTPR[index]+0.05, normFPR[index]), fontsize=15, color='black')

    plt.scatter(normTPR[min_dcf_index], normFPR[min_dcf_index], color='black', label=f"minDCF {normFPR[min_dcf_index]}")
    plt.annotate("minDCF", (normTPR[min_dcf_index]+0.05, normFPR[min_dcf_index]-0.05), fontsize=15, color='red')

    plt.xlabel('False Rejection Rate')
    plt.ylabel('False Acceptance Rate')
    plt.title('DET Curve')
    plt.show()

def objective(trial):

    history = History()

    model = get_model()

    optimizer_name = trial.suggest_categorical("optimizer", ['Adam', "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optimizers, optimizer_name)(lr=lr)

    model_compile_optuna(model, optimizer)

    for epoch in range(20):
        model.fit_generator(
            training_set,
            steps_per_epoch= training_set.samples / batch_size,
            epochs=1,
            validation_data=test_set,
            callbacks=[history]
        )

        accuracy = history.history["val_acc"][-1]

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def train(max_epoch=50, ideal_loss=0.01, ideal_accuracy=0.95, is_dnn_structure_changned=False, custom=False, compiler=0, lr=0.01):
    
    is_new_dnn = False

    history = History()

    print("=================================================")

    if (os.path.isfile(dnn_file_name_structure) and
        (os.path.isfile(dnn_file_name_weights)) and
        (is_dnn_structure_changned == False)) :

        #Load the previously trained CNN
        model = load_model()
        print("CNN is loaded")
    
    else:
        #Create a new model
        model = get_model()
        print("CNN is created")
        
        #Erase the training dynamics file if exist
        if (os.path.exists(training_dynamics_path)) :
            os.remove(training_dynamics_path)
        is_new_dnn = True
        if custom:
            if compiler == 1:
                model_compile_adam(model,lr_value=lr)
            elif compiler == 2:
                model_compile_rms(model, lr_value=lr)
            elif compiler == 3:
                model_compile_sgd(model, lr_value=lr)
        else:
            model_compile(model)
    
    ep = read_epoch() + 1

    model.fit_generator(
        training_set,
        steps_per_epoch = training_set.samples / batch_size,
        epochs = 1,
        validation_data = test_set,
        validation_steps = test_set.samples / batch_size,
        workers = 10,
        max_queue_size = 10,
        callbacks=[history]
    )

    save_training_dynamics(ep, history, with_header=is_new_dnn)

    max_val = history.history["val_acc"][-1]

    while (history.history['val_loss'][-1] >= ideal_loss and history.history["val_acc"][-1] <= ideal_accuracy):

        print("Epoch: ", ep)
        model.fit_generator(
            training_set,
            steps_per_epoch = training_set.samples / batch_size,
            epochs = 1,
            validation_data = test_set,
            validation_steps = test_set.samples / batch_size,
            workers = 10,
            max_queue_size = 10,
            callbacks=[history]
        )

        #Save the max model
        if (history.history["val_acc"][-1] > max_val and history.history["val_acc"][-1] > 0.80):
            max_val = history.history["val_acc"][-1]
            save_model(model=model, is_max_val_inclluded=True, max_val=max_val, ep=ep)
        
        #Save/Overwrite model
        save_model(model)

        ep += 1
        save_training_dynamics(ep,history, with_header=False)

        if (ep % 10 == 0):
            if (history.history['val_acc'][-1] >= ideal_accuracy):
                break

        if (history.history["val_loss"][-1] < ideal_loss):
            break

        if (ep > max_epoch):
            break

    return history


# Load X and y
training_set, test_set = get_train_test_sets()


