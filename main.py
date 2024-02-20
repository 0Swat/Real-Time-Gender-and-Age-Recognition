import cv2
import os
import re

def faceBox(faceNet, frame, return_x):
    frameHeight = frame.shape[0]  # Wysokość ramki obrazu
    frameWidth = frame.shape[1]  # Szerokość ramki obrazu

    # Tworzenie bloba z obrazu do przetwarzania przez sieć neuronową
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)  # Ustawienie bloba jako wejścia sieci
    detection = faceNet.forward()  # Przeprowadzenie detekcji twarzy
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]  # Pobranie poziomu pewności detekcji
        if confidence > 0.7:  # Filtracja detekcji o niskiej pewności
            # Obliczanie współrzędnych ramki detekcji (bounding box)
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            # Rysowanie ramki wokół wykrytej twarzy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Zwracanie ramki z narysowanymi ramkami lub samych ramkek, w zależności od argumentu return_x
    if return_x == 1:
        return frame, bboxs
    if return_x == 2:
        return bboxs

# Ścieżki do modeli wykorzystywanych do detekcji twarzy, wieku i płci
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Wczytywanie modeli za pomocą OpenCV
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-22)', '(23-34)', '(35-45)', '(46-56)', '(57-100)']
genderList = ['Male', 'Female']

def camera_age_gender():
    camera = cv2.VideoCapture(0)  # Otwarcie połączenia z kamerą
    padding = 20  # Margines dodawany do ramki detekcji twarzy

    while True:
        success, frame = camera.read()  # Odczytanie ramki z kamery
        frame, bboxs = faceBox(faceNet, frame, 1)  # Detekcja twarzy w ramce
        for bbox in bboxs:
            # Wycinanie twarzy z ramki
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            # Przygotowanie twarzy do analizy wieku i płci
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Detekcja płci
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]
            # Detekcja wieku
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]
            # Wyświetlanie wyników na ramce
            label = "{}, {}".format(gender, age)
            cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == ord(" "):  # Zakończenie pętli po naciśnięciu spacji
            break

    camera.release()
    cv2.destroyAllWindows()

def predict_age_gender(file_path):
    frame = cv2.imread(file_path)  # Wczytanie obrazu z pliku
    if frame is None:
        return (None, None)
    bboxs = faceBox(faceNet, frame, 2)  # Detekcja twarzy na obrazie
    if not bboxs:
        return (None, None)
    bbox = bboxs[0]
    # Wycinanie i przetwarzanie twarzy do analizy wieku i płci
    face = frame[max(0,bbox[1]):min(bbox[3],frame.shape[0]-1), max(0,bbox[0]):min(bbox[2], frame.shape[1]-1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    # Analiza płci i wieku
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]
    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]

    return (gender, age)

def evaluate_accuracy(folder_path):
    total_predictions = 0
    correct_predictions = 0  
    gender_only_correct = 0  
    age_only_correct = 0  
    incorrect_predictions = 0  
    print("\n")
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith('.jpg'):
            continue  
        file_path = os.path.join(folder_path, file_name)

        
        match = re.match(r"(male|female)_(\d+).jpg", file_name.lower())
        if not match:
            continue  
        real_gender, real_age = match.groups()

        
        predicted_gender, predicted_age_range = predict_age_gender(file_path)

        
        if predicted_gender is None or predicted_age_range is None:
            continue

        
        predicted_age = int(predicted_age_range.split('-')[0][1:])  
        real_age = int(real_age)
        correct_gender = real_gender.lower() == predicted_gender.lower()
        correct_age = real_age >= int(predicted_age_range.split('-')[0][1:]) and real_age <= int(predicted_age_range.split('-')[1][:-1])

        
        if correct_gender and correct_age:
            correct_predictions += 1
        elif correct_gender:
            gender_only_correct += 1
        elif correct_age:
            age_only_correct += 1
        else:
            incorrect_predictions += 1

        total_predictions += 1

        if correct_gender == True and correct_age == True:
            print(f"{file_name} Real gender: {real_gender} gender guess: {predicted_gender}\033[92m { correct_gender}\033[0m, real age: {real_age} age guess: {predicted_age_range} \033[92m{correct_age}\033[0m")
        if correct_gender == False and correct_age == True:
            print(f"{file_name} Real gender: {real_gender} gender guess: {predicted_gender}\033[91m { correct_gender}\033[0m, real age: {real_age} age guess: {predicted_age_range} \033[92m{correct_age}\033[0m")
        if correct_gender == True and correct_age == False:
            print(f"{file_name} Real gender: {real_gender} gender guess: {predicted_gender}\033[92m { correct_gender}\033[0m, real age: {real_age} age guess: {predicted_age_range} \033[91m{correct_age}\033[0m")
        if correct_gender == False and correct_age == False:
            print(f"{file_name} Real gender: {real_gender} gender guess: {predicted_gender}\033[91m { correct_gender}\033[0m, real age: {real_age} age guess: {predicted_age_range} \033[91m{correct_age}\033[0m")
    
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




