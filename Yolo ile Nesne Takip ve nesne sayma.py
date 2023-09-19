import cv2
import numpy as np
import csv
import datetime
net =cv2.dnn.readNet("C:/Users/90537/yolov3.weights", "C:/Users/90537/yolov3.cfg")
classes=[]
with open("C:/Users/90537/coco.names", "r") as f:
    classes = f.read().strip().split("\n")
csv_file=open("object_count_log.csv","w",newline="")
csv_writer=csv.writer(csv_file,delimiter=",")
csv_writer.writerow(["Label","Confidence","Object Count","Timestamp"])
cap=cv2.VideoCapture(0)
region_of_interst=[(50,50),(550,550)]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
kaydet=cv2.VideoWriter("output.avi", fourcc, 5, (640,480))

while True:
    ret,frame=cap.read()
    height,width,_=frame.shape
    blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    layer_names=net.getUnconnectedOutLayersNames()
    outs=net.forward(layer_names)
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                x=int(center_x-w/2)
                y=int(center_y-h/2)
                if region_of_interst[0][0]<x<region_of_interst[1][0] and \
                    region_of_interst[0][1]<y<region_of_interst[1][1]:
                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
    indexes=cv2.dnn.NMSBoxes(boxes,confidences ,0.5, 0.4)
    object_count=len(boxes)
    for i in range(len(boxes)):
        x,y,w,h=boxes[i]
        label = str(classes[class_ids[i]])
        confidence=confidences[i]
    
        color=(0,255,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h), color,2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f'Object Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([label, confidence, object_count, timestamp])

    # Write the frame to the output video
    kaydet.write(frame)

    # Display the frame
    cv2.imshow("Traffic Density", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects, and close CSV file
cap.release()
kaydet.release()
csv_file.close()
cv2.destroyAllWindows()
