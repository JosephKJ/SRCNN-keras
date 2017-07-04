# #
#     1   Track ID. All rows with the same ID belong to the same path.
#     2   xmin. The top left x-coordinate of the bounding box.
#     3   ymin. The top left y-coordinate of the bounding box.
#     4   xmax. The bottom right x-coordinate of the bounding box.
#     5   ymax. The bottom right y-coordinate of the bounding box.
#     6   frame. The frame that this annotation represents.
#     7   lost. If 1, the annotation is outside of the view screen.
#     8   occluded. If 1, the annotation is occluded.
#     9   generated. If 1, the annotation was automatically interpolated.
#     10  label. The label for this annotation, enclosed in quotation marks.
# #

import cv2
import numpy as np


class AnnotateFrames:

    x_max = 1500
    y_max = 1100
    classes = ['Biker', 'Pedestrian', 'Cart', 'Skater', 'Car', 'Bus']
    img = np.zeros((x_max, y_max, 3), np.uint8)
    annotation = []
    label = []

    def __init__(self, image):
        self.img = image

    def load_data(self):
        annotation_path = './annotations/bookstore/video0/annotations.txt'
        self.annotation = np.loadtxt(annotation_path, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                                     dtype=np.int)
        self.label = np.loadtxt(annotation_path, usecols=(9,) ,
                                     dtype=np.str)

    def draw_on_img(self, x1, y1, x2, y2, label):
        color = {'"Biker"':(0, 255, 255), '"Pedestrian"': (255, 0, 0), '"Cart"': (0, 0, 255), '"Skater"': (255, 255, 255),
                 '"Car"': (100, 100, 255), '"Bus"': (255, 100, 100)}
        cv2.rectangle(self.img, (x1, y1), (x2, y2), color[label], 2)

    def show_image(self):
        while True:
            cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Output Image", self.img)
            key = cv2.waitKey(10)
            if key == 27:
                break

    def annotate(self, frame_number):
        # self.img = np.zeros((self.x_max, self.y_max, 3), np.uint8)
        mask = self.annotation[:, 5] == frame_number
        data = self.annotation[mask]
        label = self.label[mask]

        if len(label) == 0:
            print('No points found')

        print("Annotating")
        for cls in self.classes:
            cls = '"' + cls + '"'
            index_arr = np.where(label == cls)
            for index in index_arr[0]:
                self.draw_on_img(data[index,1], data[index,2], data[index,3], data[index,4], cls)

        # self.show_image()
        print('Writing to file.')
        cv2.imwrite("annotated_image.png", self.img)


frame = cv2.imread('/home/dl-box/Arghya/joseph/data/StanfordDroneDataset/from_vigl_server/sdd/JPEGImages/bookstore_video0_10008.jpg')
annotator = AnnotateFrames(frame)
annotator.load_data()
annotator.annotate(10008)

