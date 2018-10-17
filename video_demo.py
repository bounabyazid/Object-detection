import time

from pydarknet import Detector, Image
import cv2

#https://towardsdatascience.com/yolov2-to-detect-your-own-objects-soccer-ball-using-darkflow-a4f98d5ce5bf
#https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html
#https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html
#http://cocodataset.org/#home
#https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/


#https://www.shutterstock.com/video
#_______________________________________________________________

#net = Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"),
#               bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0,
 #              bytes("cfg/coco.data", encoding="utf-8"))

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"),
               bytes("weights/yolov3.weights", encoding="utf-8"), 0,
               bytes("cfg/coco.data", encoding="utf-8"))

#_______________________________________________________________

def MadeVideoFile():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

    
    while True:
          r, frame = cap.read()
          cv2.imshow("preview", frame)
          out.write(frame)
          k = cv2.waitKey(1)
          if k == 0xFF & ord("q"):
             break
     
    cap.release()
    out.release()
    cv2.destroyAllWindows()
#MadeVideoFile()

#_______________________________________________________________

def LabelVideoFile():
    path = '/home/polo/.config/spyder-py3/object-detection-opencv-master/InputVideos/'
    video_path= path+'stock-footage-shenzhen-china-september-guangdong-province-cars-at-multi-lane-highway-passing.webm'
    print("Source Path:", video_path)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 20)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
    
    Objects = open('DetectedObjects/objects.tsv','w')
    Objects.write('Frames'+'\t'+'Objects'+'\t'+'Score'+'\t'+'X0'+'\t'+'Y0'+'\t'+'X'+'\t'+'Y\n')
    i = 1
    while(True):
        #print('Frame N:'+str(i))
        ret, frame = cap.read()
        
        dark_frame = Image(frame)
        results = net.detect(dark_frame)
        del dark_frame

        for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
            cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            Objects.write(str(i)+'\t'+str(cat.decode("utf-8"))+'\t'+str(score)+'\t'+str(int(x-w/2))+'\t'+str(int(y-h/2))+'\t'+str(int(x+w/2))+'\t'+str(int(y+h/2))+'\n')
        
        out.write(frame)
        i+=1
        cv2.imshow("preview", frame)
        
        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
           break

    # When everything done, release the capture
    cap.release()
    out.release()
    Objects.close()
    
    cv2.destroyAllWindows()

LabelVideoFile()
#_______________________________________________________________

def LabelVideoCam():
    #average_time = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 20)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
    
    Objects = open('DetectedObjects/objects.tsv','w')
    Objects.write('Frames'+'\t'+'Objects'+'\t'+'Score'+'\t'+'X0'+'\t'+'Y0'+'\t'+'X'+'\t'+'Y\n')
    i = 1
    while True:
          r, frame = cap.read()
          if r:
              #start_time = time.time()

              # Only measure the time taken by YOLO and API Call overhead

              dark_frame = Image(frame)
              results = net.detect(dark_frame)
              del dark_frame

              #end_time = time.time()
              #average_time = average_time * 0.8 + (end_time-start_time) * 0.2
              #print("Total Time:", end_time-start_time, ":", average_time)

              for cat, score, bounds in results:
                  x, y, w, h = bounds
                  cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                  cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                  Objects.write(str(i)+'\t'+str(cat.decode("utf-8"))+'\t'+str(score)+'\t'+str(int(x-w/2))+'\t'+str(int(y-h/2))+'\t'+str(int(x+w/2))+'\t'+str(int(y+h/2))+'\n')

              cv2.imshow("preview", frame)
              out.write(frame)
              i+=1
              
          k = cv2.waitKey(1)
          if k == 0xFF & ord("q"):
             break
     
    cap.release()
    out.release()
    Objects.close()

    cv2.destroyAllWindows()

#LabelVideoCam()