import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
# %matplotlib inline
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,  # Set the color theme for the confusion matrix
                          normalize=False,
                          name=None):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # save Picture
    plt.savefig("./picture/"+name+"_confusionmatrix.png",dpi=200)
    plt.show()


# show confusion matrix
def plot_confuse(model, x_val, y_val,labels,name):
    predict_x = model.predict(x_val)
    classes_x = np.argmax(predict_x, axis=1)
    truelabel = y_val.argmax(axis=-1)  # Convert one-hot to label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=classes_x)
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix',name=name)


# ==================================================== ============================================
# test_x is the test data, test_y is the test label (test_y is One——hot vector)
# labels is a list that stores the names of your categories, and will finally be displayed on the horizontal and vertical axes.
# labels = []
# plot_confuse(model, x_val, y_val, batch, labels, name)