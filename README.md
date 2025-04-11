# Handwritten Digit Recognition System
This system uses a Convolutional Neural Network (CNN) model to recognize handwritten digits. It includes a Jupyter
notebook that trains the model and saves it to `cnn_handwritten_recog.h5`, as well as a Python script that allows
users to draw digits using PyQt.

**Requirements**

* Python 3.x
* TensorFlow or Keras for training and saving the model
* PyQt for creating the digit drawing interface

### Setup Instructions

1.  Clone the repository: `git clone https://github.com/jcpunzalan123/handwritten-recognition.git`
2.  Install dependencies: `pip install -r requirements.txt`


**Running the Tool**
----------------------

To run the tool, execute the following command:

```python recognition_tool.py```

This will launch an interface that allows users to draw digits.
Once a digit has been drawn, hit "Enter" in keyboard to start the prediction.


# Demo

![til](https://github.com/jcpunzalan123/handwritten-recognition/blob/main/handwritten_recognition_demo.gif)
