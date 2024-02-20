# Real-Time Gender and Age Recognition

This project is a Python application that leverages OpenCV to perform real-time gender and age recognition through a webcam feed, as well as to predict gender and age from static images. The application utilizes pre-trained models for detecting faces, recognizing gender, and estimating age ranges.

## Features

- **Real-Time Gender and Age Recognition:** Utilizes a webcam to detect faces and then predict the gender and age range of each detected face in real time.
- **Static Image Analysis:** Analyzes static images to predict the gender and age range of the faces found in the images.
- **Accuracy Evaluation:** Evaluates the accuracy of gender and age predictions on a dataset of images with known attributes.

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- Pre-trained OpenCV deep learning models for face detection, gender recognition, and age estimation.

## Usage

Navigate to the project directory and run the script:  
python.py

Upon execution, the script will prompt you to choose one of the following options:

1. **Real-time gender and age recognition through webcam:** Opens your webcam and starts detecting faces, then predicts and displays the gender and age range of detected faces in real time.

2. **Test images in a folder:** Analyzes all `.jpg` images in a specified folder, predicting the gender and age range for each image and evaluating the accuracy based on the file names.

3. **Exit:** Closes the application.

## Models

Specify the paths to the downloaded models in the script:

- `faceProto`: Path to the face detector prototxt.
- `faceModel`: Path to the face detector model.
- `ageProto`: Path to the age predictor prototxt.
- `ageModel`: Path to the age predictor model.
- `genderProto`: Path to the gender predictor prototxt.
- `genderModel`: Path to the gender predictor model.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the project.

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.