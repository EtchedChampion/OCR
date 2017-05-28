# TrainAndTest.py

import cv2
import numpy as np
import operator
import os
import GenData as gd
import ContourWithData as cwd


def main():
    allContoursWithData = []  # declare empty lists,
    validContoursWithData = []  # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()  # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbersRaw = cv2.imread(".\\testData\\realPen.jpg")  # read in testing numbers image
    imgTestingNumbers = cv2.resize(imgTestingNumbersRaw, (1000, 400))

    if imgTestingNumbers is None:  # if image was not read successfully
        print "error: image not read from file \n\n"  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit function (which exits program)
    # end if

    imgThresh = gd.PreProcess(imgTestingNumbers)

    imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                              cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                              cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:  # for each contour
        contourWithData = cwd.ContourWithData(npaContour)  # instantiate a contour with data object
        allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:  # for all contours
        if gd.CheckIfContourIsValid(contourWithData):  # check if valid
            validContoursWithData.append(contourWithData)  # if so, append to valid contour list
            # end if
    # end for

    validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right

    strFinalString = ""  # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:  # for each contour
        # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,  # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth,
                       contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
                      (0, 255, 0),  # green
                      2)  # thickness

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                 # crop char out of threshold image
                 contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (gd.RESIZED_IMAGE_WIDTH,
                                            gd.RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape(
            (1, gd.RESIZED_IMAGE_WIDTH * gd.RESIZED_IMAGE_HEIGHT))  # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                     k=2)  # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

        strFinalString = strFinalString + strCurrentChar  # append current char to full string
    # end for

    print "\n" + strFinalString + "\n"  # show the full string

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)  # show input image with green boxes drawn around found digits
    cv2.waitKey(0)  # wait for user key press

    cv2.destroyAllWindows()  # remove windows from memory

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
