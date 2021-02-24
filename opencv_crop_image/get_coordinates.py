import cv2
import numpy as np

class GetCoordinates:

    # Keys to exit or reset drawing
    EXIT_GUI = ord("e")
    RESET_GUI = ord("r")

    # output: the output file contains coordinates of drawn boxes
    # caption: a title of the gui window
    # image: a path to the image file
    # click_counter: To count the number of mouse clicks while drawing a box. 
    #                Maximum is 4 to draw one box.
    # space_id: To label each drawn box with a number
    # coordinates: An array to store coordinates of one box.
    def __init__(self, image, output):
        self.image = cv2.imread(image).copy()
        self.output = output
        self.caption = image
        self.click_counter = 0
        self.space_id = 0
        self.coordinates = []

        # Create a window to display the image and bind the callback function to window
        cv2.namedWindow(self.caption, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.caption, self.__drawing_a_box)

    # source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
    # Mouse call back function
    def __drawing_a_box(self, event, x, y, flags, params):
        # on left mousse click event
        if event == cv2.EVENT_LBUTTONDOWN:
            # add this point to coordinates array
            self.coordinates.append((x, y))
            # count the click
            self.click_counter += 1
            # One box is done drawing = 4 clicks
            if self.click_counter >= 4:
                self.__drawn_is_done_event()
            # User still making more points to make a box
            elif self.click_counter > 1:
                self.__is_drawing_event()

        cv2.imshow(self.caption, self.image)


    # draw a line by connecting two points while user is making points
    # source: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html?highlight=line#cv2.line
    def __is_drawing_event(self):
        cv2.line(self.image, self.coordinates[-2], self.coordinates[-1], (255, 0, 0), 4)

    # connects remaining lines when 4 points are created
    def __drawn_is_done_event(self):
        cv2.line(self.image,
                     self.coordinates[2],
                     self.coordinates[3],
                     (255, 0, 0),
                     4)
        cv2.line(self.image,
                     self.coordinates[3],
                     self.coordinates[0],
                     (255, 0, 0),
                     4)

        # Write the coordinates of this parking space with its label id to the output
        self.output.write("- space_id: " + str(self.space_id) + "\n  points:\n" +
                          "  - [" + str(self.coordinates[0][0]) + ", " + str(self.coordinates[0][1]) + "]\n" +
                          "  - [" + str(self.coordinates[1][0]) + ", " + str(self.coordinates[1][1]) + "]\n" +
                          "  - [" + str(self.coordinates[2][0]) + ", " + str(self.coordinates[2][1]) + "]\n" +
                          "  - [" + str(self.coordinates[3][0]) + ", " + str(self.coordinates[3][1]) + "]\n")
        print('after writing outut')
        print(self.output)
        # draw the final contours outlines using drawContours functions
        contours = np.array(self.coordinates)
        self.__draw_contours(self.image, contours, str(self.space_id + 1))

        # reset the coordinates array
        for i in range(0, 4):
            self.coordinates.pop()

        # increment space_id
        self.space_id += 1
        # reset click counter
        self.click_counter = 0

        x, y = [], []
        index = 0
        for contour_line in contours:
            for contour in contour_line:
                if index % 2 == 0:
                    x.append(contour)
                else:
                    y.append(contour)
                index+=1
        #         x.append(contour[0][0])
        #         y.append(contour[0][1])

        x1, x2, y1, y2 = min(x), max(x), min(y), max(y)

        # print('[y1 = %f, y2 = %f, x1 = %f, x2=%f' %(y1, y2, x1, x2))
        # cropped = self.image[y1:y2, x1:x2]
        # cv2.imshow("crop" + str(self.space_id), cropped)
        # cv2.imwrite("saved_crop/crop" + str(self.space_id)+".jpg", cropped)



    def __draw_contours(self, image, contours, space_id):

        # Refer the parameters meaning in the documentation
        # source: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html?highlight=line#cv2.line
        cv2.drawContours(image,
                            [contours],
                            contourIdx=-1,
                            color=(255, 0, 0),
                            thickness=4,
                            lineType=cv2.LINE_8)
        
        # Fun math to compute center of the contours
        # source: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
        get_moments = cv2.moments(contours)
        center_position = (int(get_moments["m10"] / get_moments["m00"]) - 3,
                int(get_moments["m01"] / get_moments["m00"]) + 3)

        #Put the space id in the center of the box
        #source: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html?highlight=line#cv2.line
        cv2.putText(image,
                        space_id,   # text to be drawn
                        center_position,     # position
                        cv2.FONT_HERSHEY_SIMPLEX, # font
                        1.0,        # fontScale
                        (0, 255, 0),    # color
                        2,                  # thickness
                        cv2.LINE_AA)    # line type


    # Listen to a matching key to quit or reset
    def finish_segmenting(self):
        while True:
            cv2.imshow(self.caption, self.image)
            key = cv2.waitKey(0)

            if key == GetCoordinates.RESET_GUI:
                self.image = self.image.copy()
            elif key == GetCoordinates.EXIT_GUI:
                break
        cv2.destroyWindow(self.caption)