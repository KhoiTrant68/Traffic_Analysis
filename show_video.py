import cv2
import imutils


def show_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=640, width=640)
        if ret:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "Highway_Management/output.mp4"
    show_video(path)
