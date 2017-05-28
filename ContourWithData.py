import cv2


class ContourWithData:
    def __init__(self, contour):
        self.npaContour = contour  # assign contour to contour with data
        self.boundingRect = cv2.boundingRect(contour)  # get the bounding rect
        self.CalculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
        self.fltArea = cv2.contourArea(contour)  # calculate the contour area
        pass

    npaContour = None  # contour
    boundingRect = None  # bounding rect for contour
    intRectX = 0  # bounding rect top left corner x location
    intRectY = 0  # bounding rect top left corner y location
    intRectWidth = 0  # bounding rect width
    intRectHeight = 0  # bounding rect height
    fltArea = 0.0  # area of contour

    def CalculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth + 2
        self.intRectHeight = intHeight + 2
