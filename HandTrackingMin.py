import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #Camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # tracking configurations press ctrl+click on Hands() for more
mpDraw = mp.solutions.drawing_utils    #Draw the hand points

pTime = 0
cTime = 0

while True:
    success, img = cap.read()  #frame
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # converts img into RGB because 'hands' onl take RGB images
    results = hands.process(imgRGB)  # process frame and give result
    #print(results.multi_hand_landmarks)  # print if hand present in frame five landmark value if not give NONE

    if results.multi_hand_landmarks:      # for multiple hands
        for hanndLms in results.multi_hand_landmarks:
            for id,lm in enumerate(hanndLms.landmark):   # for coordinates of hands in frame(x,y,z)
               # print(id,lm)
                h,w,c = img.shape  #height width
                cx, cy = int(lm.x*w),int(lm.y*h)     #position of center
                print(id,cx,cy)
                #if id ==0:  # highlighting point on hand to do work (out of 21 pts)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img,hanndLms,mpHands.HAND_CONNECTIONS)   #create hand lines to points

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)    #show fps

    cv2.imshow("Image",img)     #run webcam
    cv2.waitKey(1)
