import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == ord(" "):
        break

camera.release()
cv2.destroyAllWindows()
