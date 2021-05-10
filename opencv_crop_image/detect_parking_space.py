import cv2
import numpy as np
from classifier import classifier
import pandas as pd
import urllib3
import json



class DetectSpaces:
    DETECT_DELAY = 1
    LAPLACIAN = 2.0
    url = 'http://localhost:8080/data'

    def __init__(self, video, points, start_frame):
        self.video = video
        self.park_spaces_data = points
        self.start_frame = start_frame
        self.contours_array = []
        self.bounding_array = []
        self.mask_array = []

    def detect_spaces(self):
        # capture video frames
        video_capture = cv2.VideoCapture(self.video)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        park_spaces_data = self.park_spaces_data
  

        empty_result = []
        occupied_result = []

   

        # Loading drawn boxes on the video frame
        for one_space_data in park_spaces_data:
            # open points of one parking space as a numpy array
            curr_points = np.array(one_space_data["points"]) 
            # bounding of the contour
            bounding_rect = cv2.boundingRect(curr_points)

            # shift contour to ROI to speed up calculation
            new_points = curr_points.copy()
            new_points[:, 0] = curr_points[:, 0] - bounding_rect[0]
            new_points[:, 1] = curr_points[:, 1] - bounding_rect[1]
            
            self.contours_array.append(curr_points)
            self.bounding_array.append(bounding_rect)

            mask = cv2.drawContours(np.zeros((bounding_rect[3], bounding_rect[2]), dtype=np.uint8),
                [new_points],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=cv2.LINE_8)

            mask = mask == 255
            self.mask_array.append(mask)
        
        # Initialize status of all marked spaces as Occupied (False)
        parking_status_array = [False] * len(park_spaces_data)
        buffer = [None] * len(park_spaces_data)

        init_pos = 0
        start_video = True

        while video_capture.isOpened():
            # Get current position of the video in seconds
            current_position_secs = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # Read frame
            result, init_frame = video_capture.read()
            # Remove the background, size 5x5 dimension 3
            blurred_frame = cv2.GaussianBlur(init_frame.copy(), (5,5), 3)
            # Convert the image color to RGB
            #rgb_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB)
            new_frame = init_frame.copy()
            grey_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
            #classify parking status 
            #(thresh, blackAndWhiteImage) = cv2.threshold(grey_frame, 127, 255, cv2.THRESH_BINARY)
            ready_data = []
            if start_video or (int(current_position_secs) > init_pos and init_pos % 10 == 0):
                if int(current_position_secs) > init_pos:
                    print('Time to classify ' + str(int(current_position_secs)) + '- ' + str(init_pos))
                
                print(current_position_secs)
                print(init_pos)
                print('---------------------')
                for index, park_space in enumerate(park_spaces_data):
                    curr_points = np.array(park_space["points"])
                    bounding_rect = self.bounding_array[index]

                    new_points[:, 0] = curr_points[:, 0] - bounding_rect[0]
                    new_points[:, 1] = curr_points[:, 1] - bounding_rect[1]
                    roi_frame = init_frame[bounding_rect[1]:(bounding_rect[1] + bounding_rect[3]), bounding_rect[0]:(bounding_rect[0] + bounding_rect[2])]
                    roi_gray_frame = grey_frame[bounding_rect[1]:(bounding_rect[1] + bounding_rect[3]), bounding_rect[0]:(bounding_rect[0] + bounding_rect[2])]
              
                    ########## TEST ##############
                   
              
                    
                   
                    #image = self.center_crop(roi_gray_frame, [100, 100])
               
            
                    #status = self.getResult(modified_image)
                    ##############################
                    
                    #Run classfier model
                    score, status = classifier(roi_frame)

                    # if start_video:
                    #     if status:
                    #         empty_result.append([index+1, status, score])
                    #     else:
                    #         occupied_result.append([index+1, status, score])
                    res_status = 0
                    if status:
                        res_status = 1
                    ready_data.append({"name": 'L' + str(index+1), "location":"Library", "status": status , 'reserve_status': res_status})
                    # if score < 0.6:
                    #     laplacian = cv2.Laplacian(roi_gray_frame, cv2.CV_64F)
                    #     status_lap = np.mean(np.abs(laplacian * self.mask_array[index])) < DetectSpaces.LAPLACIAN    
                    #     if status_lap != status:
                    #         status = status_lap
                    
                    if status:
                        print(index+1)

                    #print("Space ID: " +str(index +1 ) + "; Status = " + str(status) + "; Value: " + str(score)) #str(np.mean(np.abs(laplacian * self.mask_array[index]))))
                    # Save the current time if status is changed
                    if status != parking_status_array[index] and buffer[index] == None:
                        buffer[index] = current_position_secs
                    # Status is different and detection is delaying
                    elif status != parking_status_array[index] and buffer[index] != None:
                        if current_position_secs - buffer[index] > DetectSpaces.DETECT_DELAY:
                            parking_status_array[index] = status
                            buffer[index] = None
                        continue
                    # Status is not changed
                    elif status == parking_status_array[index] and buffer[index] != None:
                        buffer[index] = None
                        continue
                    
                    #cv2.imwrite("saved_roi/roi" + str(index+1)+".jpg", roi_frame)

                correct_payload = json.dumps(ready_data).encode('utf-8')
                http = urllib3.PoolManager()
                r = http.request('POST', DetectSpaces.url, headers={'Content-Type': 'application/json'}, body=correct_payload)
                # r = http.request('GET', DetectSpaces.url)
                # print("\ndata from server for get request\n")
                # print(r.data)
                # Change the color of the boxes if their status changed
            for index, park_space in enumerate(park_spaces_data):
          
                curr_points = np.array(park_space["points"]) 
                    
                color = (0, 255, 0) if parking_status_array[index] else (0, 0, 255)
                    
                    # Redraw and change the color of the contour

                cv2.drawContours(new_frame,
                            [curr_points],-1,color,2,cv2.LINE_8)
                moments = cv2.moments(curr_points)

                center = (int(moments["m10"] / moments["m00"]) - 3,int(moments["m01"] / moments["m00"]) + 3)

                cv2.putText(new_frame,
                        str(park_space["space_id"] + 1),
                        center,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,255),
                        1,
                        cv2.LINE_AA)
            

            
            
            
            init_pos = int(current_position_secs)

            start_video = False
            
            cv2.imshow(str(self.video), new_frame)
            key = cv2.waitKey(1)
            if key == ord("e"):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Number of empty spaces detected: ' + str(len(empty_result)))
        # print('Number of occupied spaces detected: ' + str(len(occupied_result)))
        # df_occ = pd.DataFrame(occupied_result)
        # df_emp = pd.DataFrame(empty_result)

        # df_occ.to_csv("test_data/occupied_pc.csv", header=['Space Id', 'Status', 'Score'], index=False)
        # df_emp.to_csv("test_data/empty_pc.csv", header=['Space Id', 'Status', 'Score'], index=False)
        

    def center_crop(self, img, dim):
        width, height = img.shape[1], img.shape[0]
        print(width)
        print(height)
        #process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img

    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    
    def isLightOrDark(self, rgbColor):
        [r,g,b]=rgbColor
        hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
        if (hsp>127.5):
            return 'light'
        else:
            return 'dark'

    def getResult(self, blackAndWhiteImage):
        clf = KMeans(n_clusters = 2)
        labels = clf.fit_predict(blackAndWhiteImage)
        counts = Counter(labels)
        #results_labels = [filename]
        #print(counts)

        center_colors = clf.cluster_centers_
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
        # print(filename)

        # print(rgb_colors)
        # print(isLightOrDark(rgb_colors[0]))
        # print(isLightOrDark(rgb_colors[1]))
        p1 = counts[0] / (counts[0] + counts[1])
        p2 = 1 - p1
        # print(p1)
        # print(p2)
        # print(counts)
        if (p1 > p2):
            if (p1 > 0.9) or (self.isLightOrDark(rgb_colors[0]) == self.isLightOrDark(rgb_colors[1])):
                
                return True
            elif p1 >= 0.5 :
                
                return False
        else:
            if (p2 > 0.9) or (self.isLightOrDark(rgb_colors[0]) == self.isLightOrDark(rgb_colors[1])):
      
                return True
            elif p2 >= 0.5 :
     
                return False
        #plt.figure(figsize = (8, 6))

        #plt.pie(counts.values(), labels = results_labels, colors = hex_colors)
        return False