import cv2
import numpy as np
import time
import autopy
import handTrackingModule as htm

wCam,hCam = 640, 480
frameR = 70
smoothener = 5
pLocx , pLocy = 0,0
cLocx , cLocy = 0,0
ptime = 0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detectr = htm.handDetector(maxHands=1)
wScr , hScr = autopy.screen.size()
# print(wScr , hScr)

while True:
    # 1.Find hand landmarks
    success,img = cap.read()
    img = detectr.findHands(img)
    lmList , bbox = detectr.findPosition(img)

    #2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)

    #3. Check which fingers are up
    fingers = detectr.fingersUp()
    cv2.rectangle(img , (frameR,frameR),(wCam-frameR , hCam-frameR),(255,0,255),2)

    #4. Only Index finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
    #     # Convert coordinates
        x3 = np.interp(x1 , (frameR,wCam-frameR),(0,wScr))
        y3 = np.interp(y1 , (frameR,hCam-frameR), (0,hScr))

        #6.Smoothen values 
        cLocx = pLocx + (x3 - pLocx)/smoothener
        cLocy = pLocy + (y3 - pLocy)/smoothener  
        
        #7. Move mouse
        autopy.mouse.move(wScr-cLocx,cLocy)
        cv2.circle(img , (x1,y1) , 15 , (255,0,255),cv2.FILLED)
        pLocx , pLocy = cLocx , cLocy
    
    #8. Both index and middle fingers are up: Cicking mode
    if fingers[1] == 1 and fingers[2] == 1:
        length ,img ,info = detectr.findDistance(8,12,img)
        print(length)
        if length<40:
            cv2.circle(img, (info[4],info[5]),15,(0,255,255),cv2.FILLED)
            autopy.mouse.click()
    #9. Frame rate
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN ,2,(255,0,0),2 )
    
    cv2.imshow("Image",img)

    if cv2.waitKey(1)==ord('q'):
        break
    

