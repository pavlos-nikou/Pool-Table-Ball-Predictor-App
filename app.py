from turtle import circle
from types import NoneType
import numpy as np
import cv2

cropX = 75
cropY = 50

def detectCueStick(frame,processedFrame):
    gray = cv2.cvtColor(processedFrame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    return frame,lines


def detectBalls(frame, processedFrame):
    mask = cv2.inRange(processedFrame,(0,45,0),(100,255,120))
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 10, param1=800, param2=10, minRadius=10, maxRadius=20)
    if type(circles) == NoneType:
        # print(type(circles))
        return frame,[0]
    # print(circles)
    circles = circles.astype("int")
    # print(circles[0, :])
    for i in circles[0, :]:
        # print(i)
        cv2.circle(frame, (i[0] + cropY, i[1]+ cropX), i[2], (0, 255, 0), 2)
        cv2.circle(frame, (i[0] + cropY, i[1]+ cropX), 2, (0, 0, 255), 3)
    cv2.imshow("mask",mask)
    return frame,circles

def crop(frame):
    croppedFrame = frame[cropX:650,cropY:1250]
    return croppedFrame

if __name__ == "__main__":
    cap = cv2.VideoCapture("poolSample1.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        cropedFrame = crop(frame)   
        # greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(cropedFrame, (51, 51), 0)
        RGBframe = cv2.cvtColor(blurFrame,cv2.COLOR_BGR2RGB)
        # mainFrame,balls = detectBalls(frame,RGBframe)
        mainFrame,lines = detectCueStick(frame,RGBframe)
        print(lines)
        cv2.imshow("video", mainFrame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()