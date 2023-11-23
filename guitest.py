
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import sys


class MyWindow(QMainWindow):
    def __init__(self):
         super(MyWindow, self).__init__()
         self.setGeometry(200,200,300,300)
         self.setWindowTitle("Durian Classification")
         self.initUI()
    
    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Hello World")
        self.label.move(50,50)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Select Image")
        self.b1.clicked.connect(self.selectImage)

        self.b2 = QtWidgets.QPushButton(self)
        self.b2.setText("Predict Image")
        self.b2.clicked.connect(self.predictImage)

    def selectImage(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)") 
        if file_path:
            img = Image.open(file_path)
            my_image = ImageOps.fit(img, (128,128))
            global my_image_re
            my_image_re = tf.keras.applications.vgg16.preprocess_input(np.array(my_image))
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(img)
            axarr[0].set_title('Original Image')
            axarr[1].imshow(my_image)
            axarr[1].set_title('Resized Image')
            axarr[2].imshow(my_image_re)
            axarr[2].set_title('Preprocessed Image')
            plt.show()

    def predictImage(self):
        try:    
            cnn_model = load_model('C:\\Users\\steph\\OneDrive\\Documents\\Code Practice\\DurianClassification\\durian_classification_trained_model.h5')
            cnn_model.run_eagerly=True
            probabilities = cnn_model.predict(np.array([my_image_re,]), verbose=0)[0,:]
            print(probabilities)
            number_to_class = ['D13','D24','D197']
            index = np.argsort(probabilities)
            predictions = {
                "class1":number_to_class[index[2]],
                "class2":number_to_class[index[1]],
                "class3":number_to_class[index[0]],
                "prob1":probabilities[index[2]],
                "prob2":probabilities[index[1]],
                "prob3":probabilities[index[0]],
                }
            prediction_text = f"Class 1: {predictions['class1']} ({predictions['prob1']:.2f})\n" \
                            f"Class 2: {predictions['class2']} ({predictions['prob2']:.2f})\n" \
                            f"Class 3: {predictions['class3']} ({predictions['prob3']:.2f})"
            prediction_label = QtWidgets.QLabel(self)
            prediction_label.setText(prediction_text)
            prediction_label.move(50,50)
            prediction_label.show()
        except NameError:
            error_label = QtWidgets.QLabel(self)
            error_label.setText("Please select an image first")
            error_label.move(50,50)
            error_label.show()
        except Exception:
            error_label = QtWidgets.QLabel(self)
            error_label.setText("Something went wrong, please try again")
            error_label.move(50,50)
            error_label.show()

                

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()
