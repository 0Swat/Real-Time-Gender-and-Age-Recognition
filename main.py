import cv2
import os
import re

def faceBox(faceNet, frame, return_x):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if return_x == 1:
        return frame, bboxs
    if return_x == 2:
        return bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def camera_age_gender():
    camera = cv2.VideoCapture(0)
    padding = 20

    while True:
        success, frame = camera.read()
        frame, bboxs = faceBox(faceNet, frame, 1)
        for bbox in bboxs:
            #face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]


            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            label = "{}, {}".format(gender, age)
            cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == ord(" "):
            break

    camera.release()
    cv2.destroyAllWindows()

def predict_age_gender(file_path):
    frame = cv2.imread(file_path)
    if frame is None:
        return (None, None)
    bboxs = faceBox(faceNet, frame, 2)
    if not bboxs:
        return (None, None)  # No face detected
    bbox = bboxs[0]  # Taking the first detected face
    face = frame[max(0,bbox[1]):min(bbox[3],frame.shape[0]-1), max(0,bbox[0]):min(bbox[2], frame.shape[1]-1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]

    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]

    return (gender, age)

def evaluate_accuracy(folder_path):
    total_predictions = 0
    correct_predictions = 0  # Both gender and age correct
    gender_only_correct = 0  # Only gender correct
    age_only_correct = 0  # Only age correct
    incorrect_predictions = 0  # Neither gender nor age correct
    print("\n")
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith('.jpg'):
            continue  # Skip non-JPG files

        file_path = os.path.join(folder_path, file_name)

        # Extract real gender and age from the file name
        match = re.match(r"(male|female)_(\d+).jpg", file_name.lower())
        if not match:
            continue  # Skip files not matching the naming convention
        real_gender, real_age = match.groups()

        # Predict gender and age range
        predicted_gender, predicted_age_range = predict_age_gender(file_path)

        # Skip if no face detected
        if predicted_gender is None or predicted_age_range is None:
            continue

        # Determine correctness of predictions
        predicted_age = int(predicted_age_range.split('-')[0][1:])  # Taking the lower bound of the predicted age range
        real_age = int(real_age)
        correct_gender = real_gender.lower() == predicted_gender.lower()
        correct_age = real_age >= int(predicted_age_range.split('-')[0][1:]) and real_age <= int(predicted_age_range.split('-')[1][:-1])

        # Categorize the prediction
        if correct_gender and correct_age:
            correct_predictions += 1
        elif correct_gender:
            gender_only_correct += 1
        elif correct_age:
            age_only_correct += 1
        else:
            incorrect_predictions += 1

        total_predictions += 1

        print(f"{file_name}  gender guess: {correct_gender}, age guess: {correct_age}")
    
    net_accuracy = ((correct_predictions + age_only_correct*0.5 + gender_only_correct*0.5) / total_predictions) * 100
    age_accuracy = ((correct_predictions + age_only_correct) / total_predictions) * 100
    gender_accuracy = ((correct_predictions + gender_only_correct) / total_predictions) * 100

    return {
        'total': total_predictions,
        'correct': correct_predictions,
        'gender_only': gender_only_correct,
        'age_only': age_only_correct,
        'incorrect': incorrect_predictions,
        'net_accu': net_accuracy,
        'age_accu': age_accuracy,
        'gender_accu': gender_accuracy
    }

def print_results(results):
    print("\nResults:")
    print(f"Total images: {results['total']}")
    print(f"Correctly recognized (both gender and age): {results['correct']}")
    print(f"Correctly recognized gender only: {results['gender_only']}")
    print(f"Correctly recognized age only: {results['age_only']}")
    print(f"Incorrectly recognized: {results['incorrect']}")
    print(f"Net accuracy: {results['net_accu']} %")
    print(f"Age accuracy: {results['age_accu']} %")
    print(f"Gender accuracy: {results['gender_accu']} %")


def main_menu():
    choice = input("Choose an option: \n 1. Real-time gender and age recognition through webcam \n 2. Test images in a folder \n 3. Exit \n > ")
    return choice

choice = main_menu()

if choice == '1':
    camera_age_gender()
elif choice == '2':
    folder_path = input("Enter the folder path: ")
    results = evaluate_accuracy(folder_path)
    print_results(results)
elif choice == '3':
    print("Exiting...")
else:
    print("Invalid option.")




