# Functions/utils used in the notebooks
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# Model performance metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def download_stamp(oid, client, output_dir, to_delete):
    """
    Download a stamp from the ALeRCE cliente based on the 
    oid provided. If the oid provided present a exception, 
    it's saved in a 'to_delete' list to delete it on the dataset

    """
    file_path = os.path.join(output_dir, f"{oid}_stamps.fits")

    if os.path.exists(file_path):
        return True  # Ya estaba descargado

    try:
        stamps = client.get_stamps(oid)
        stamps.writeto(file_path, overwrite=True)
        time.sleep(0.05)
        return True

    except:
        to_delete.append(oid) 
        return False


def visual_fits(path):
    '''
    """
    Helper function for visualizing the three
    images stored in a .fits stamp file.
    """
    '''
    hdu = fits.open(path)

    science = hdu[0].data
    reference = hdu[1].data
    difference = hdu[2].data

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

    # sCIENCE
    axes[0].imshow(science, cmap="gray")
    axes[0].set_title("Science")
    axes[0].axis('off') 

    # REFERENCE
    axes[1].imshow(reference, cmap="gray")
    axes[1].set_title("Reference")
    axes[1].axis('off') 

    # DIFFERENCE
    axes[2].imshow(difference, cmap="gray")
    axes[2].set_title("Difference")
    axes[2].axis('off') 

    plt.show()
    hdu.close()


def clean_labels(labels, to_delete, output_dir):
    '''
    """
    Helper function that continues the corrupted
    OID filtering process started during the
    download. It appends to `to_delete` any
    entries whose stamps do not have the expected
    shape (63, 63, 3) and returns the labels.
    '''
    for idx, row in labels.iterrows():
        oid = row['oid']
        label = row['label']

        if oid not in to_delete:
            hdu = fits.open(os.path.join(output_dir, f"{oid}_stamps.fits"))
            stamp = np.stack([hdu[0].data, hdu[1].data, hdu[2].data], axis=-1)

            if stamp.shape != (63,63,3):
                to_delete.append(oid)

    labels_cleaned = labels[~labels['oid'].isin(to_delete)].reset_index(drop=True)
    return labels_cleaned


def build_Xy(labels_clean, output_dir):
    '''
    Helper function that build the X and y sets
    to train a neuronal network
    '''
    X = []
    y = []

    for idx, row in labels_clean.iterrows():
        oid = row['oid']
        label = row['label']

        path = os.path.join(output_dir, f"{oid}_stamps.fits")
        hdu = fits.open(path)

        sci = hdu[0].data
        ref = hdu[1].data
        diff = hdu[2].data

        stamp = np.stack([sci, ref, diff], axis=-1)  # (63,63,3)
        X.append(stamp)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def normalize_img(X):
    """
    Apply L_2 normalization to images (as done in braai) 
    to ensure proper processing by the CNN
    """
    X_norm = np.asarray(X, dtype=np.float32) # overflow to 0
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X_norm, axis=(1, 2), keepdims=True)
    norms = np.where(norms == 0, 1, norms)

    X_norm /= norms
    
    return X_norm


def evaluate_cnn_model(model, X_test, y_test, class_names=['real', 'bogus']):
    """
    Helper function which evaluates a CNN model. 
    Its show the confussion matrix and classification report
    """
    # model prediction
    y_prob = model.predict(X_test)

    # This block is used for the multiclass extension in notebook III
    if y_prob.shape[1] == 1:
        y_pred = (y_prob > 0.5).astype(int).flatten() # threshold majority
    else:
        y_pred = np.argmax(y_prob, axis=1) # multiclass

    y_true = y_test

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', colorbar=True)
    plt.title("Confussion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    # (f1-score, recall, precision)
    print("\n               ===== Classification report =====")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def plot_training_history(history):
    """
    Helper function for ploting the loss and accuracy
    between validation and training set 
    """
    plt.figure(figsize=(12, 5))

    # LOSS
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.legend()

    # ACCURACY
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    plt.show()