# GenData.py

import sys
import numpy as np
import cv2
import os
import ContourWithData as cwd
import operator

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 40  # was 25 @ 11/05/2017

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


###################################################################################################


def ListFolders(path):
    r = []
    subPaths = [x[0] for x in os.walk(path)]
    for subdir in subPaths:
        folders = os.walk(subdir).next()[1]
        if len(folders) > 0:
            for folder in folders:
                r.append(subdir + "/" + folder)
    return r


def CheckIfContourIsValid(self):  # this is oversimplified, for a production grade program
    if self.fltArea < MIN_CONTOUR_AREA: return False  # much better validity checking would be necessary
    return True


def PreProcess(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      255,  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV, # invert so foreground will be white, background will be black
                                      11,  # size of a pixel neighborhood used to calculate threshold value
                                      2)  # constant subtracted from the mean or weighted mean

    cv2.imwrite(".\\tmpDump/orgImage.png", image)
    cv2.imwrite(".\\tmpDump/grayImage.png", imgGray)
    cv2.imwrite(".\\tmpDump/blurredImage.png", imgBlurred)
    cv2.imwrite(".\\tmpDump/imgTresh.png", imgThresh)


    return imgThresh


def FindValidContours(image):
    imgThresh = PreProcess(image)

    #cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference

    imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                              cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                              cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

    allContoursWithData = []  # declare empty lists,
    validContoursWithData = []  # we will fill these shortly
    for npaContour in npaContours:  # for each contour
        contour = cwd.ContourWithData(npaContour)
        allContoursWithData.append(contour)  # add contour with data object to list of all contours with data

    for contourWithData in allContoursWithData:  # for all contours
        if CheckIfContourIsValid(contourWithData):  # check if valid
            validContoursWithData.append(contourWithData)  # if so, append to valid contour list
            # end if
    # end for

    validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right
    return validContoursWithData, imgThresh


def AnalyzeContours(contour, image, imgThresh):
        # draw rectangle around each contour as we ask user for input
        cv2.rectangle(image,  # draw rectangle on original training image
                      (contour.intRectX, contour.intRectY),  # upper left corner
                      (contour.intRectX + contour.intRectWidth, contour.intRectY + contour.intRectHeight),
                      # lower right corner
                      (0, 0, 255),  # red
                      2)  # thickness

        imgROI = imgThresh[contour.intRectY: contour.intRectY + contour.intRectHeight,
                 contour.intRectX: contour.intRectX + contour.intRectWidth]  # crop char out of threshold image
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                            RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

        # cv2.imwrite(".\\tmpDump/contourNormal" + str(contour.intRectX) + ".png", imgROI)
        cv2.imwrite(".\\tmpDump/contourResized" + str(contour.intRectX) + ".png", imgROIResized)

        return imgROIResized


def ProcessImage(image, char, npaFlattenedImages, intClassifications):
    intChar = ord(char)
    if image is None:  # if image was not read successfully
        print "error: image not read from file \n\n"  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit function (which exits program)
    # end if

    validContours, imgThresh = FindValidContours(image)

    for contour in validContours:
        imgROIResized = AnalyzeContours(contour, image, imgThresh)

        intClassifications.append(intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)

        npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
        npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)  # add current flattened impage numpy array to list of flattened image numpy arrays

    cv2.imwrite(".\\tmpDump/TotalContour.png", image)

    return npaFlattenedImages, intClassifications


def startImage(imagePath, char, npaFlattenedImages, intClassifications):
    image = cv2.imread(imagePath)  # read in training numbers image
    npaFlattenedImages, intClassifications = ProcessImage(image, char, npaFlattenedImages, intClassifications)
    # cv2.imwrite(char + "_done.png", image)
    return npaFlattenedImages, intClassifications


def TrainFolder(path, npaFlattenedImages, intClassifications):
    list = os.listdir(path)
    char = os.path.basename(path)
    print "starting folder: " + char
    for file in list:
        print "Analyzing " + file
        file = path + "\\" + file
        npaFlattenedImages, intClassifications = startImage(file, char, npaFlattenedImages, intClassifications)

    return npaFlattenedImages, intClassifications


def main():
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    intClassifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

    for folder in ListFolders('.\\debug'): # ListFolders('.\\TrainingSet'):
        npaFlattenedImages, intClassifications = TrainFolder(folder, npaFlattenedImages, intClassifications)

    fltClassifications = np.array(intClassifications,
                                  np.float32)  # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

    print "\n\ntraining complete !!\n"

    np.savetxt("classifications.txt", npaClassifications)  # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)  #

    cv2.destroyAllWindows()  # remove windows from memory

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
