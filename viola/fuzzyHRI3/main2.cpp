#include <gustereotools/planviewmaps.h>
#include <gustereotools/peopletracker.h>
#include <gustereotools/stereofiller8lines.h>
#include <gustereotools/personmaskextractor.h>
#include <gustereotools/stereogaussbackground.h>
#include <gustereo3/stereosetvideoreader.h>
#include <gustereo3/_3dtransformationmatrix.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <guopencv/utilities.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
//#include <guavi/guaviwrite.h>
#include <gu/gustringutils.h>
#include <guopencv/haarobjectdetector.h>
#include <guia/fuzzymatlab.h>
#include <imagecomposer.h>
#include <smileestimator.h>

using namespace guopencv;
using namespace guia_addon;
using namespace gustereotools::planviewmaps;
using namespace gustereotools::pv_peopletrackers;
using namespace gustereotools::background;
using namespace gustereotools::shapeFromStereo;
using namespace gustereo3;

#define LEARNINGRATESMILE 0.3      // learning rate for smile estimation
#define LEARNINGRATEATTENTION 0.3  // learning rate for attention estimation ***
#define MAXIMUMPERSONS \
  10  // maximum number of people that can be tracked at the same time
#define MOUTHCENTER \
  0.1  // We consider that the mouth center should be 10% left/right and up/down
       // from the down half of the face square
#define ERRORDISTANCEMETERS \
  0.25  // we consider this the maximum error distance in meters which to
        // consider that an openCV face belongs to the tracked pers
#define ERRORDISTANCEPIXELS \
  20  // we consider this the maximum error distance in pixels which to consider
      // that an openCV face belongs to the tracked pers
#define FRAMESPERPERSON 5  // number of pixels to record each person's position
#define SPEED 30              // distance to consider that a person is moving
#define HEADERROR 10          // pixels below and aside to search for a head
#define PIXELSLINE 3          // pixels in a row to be considered a face border
#define LEARNINGRATEARMS 0.3  // learning rate for arms movement ***
#define GESTUREMINIMUMX 20  // person's minimum gesture width
#define GESTUREMAXIMUMX 35  // person's maximum gesture width
#define GESTUREMINIMUMY 50  // person's minimum gesture height
#define GESTUREMAXIMUMY 70  // person's maximum gesture height
#define CENTERBINARYPICTUREX 60  // pixel center x of binary picture
#define CENTERBINARYPICTUREY -70  // pixel center y of binary picture
#define UMBRALDISTANCEGESTURE \
  0.9  // distance of the pixel to the center of person to consider that it
       // could belong to the extending arm
#define MINDISTANCE 0.5  // minimum distance of person to camera
#define MAXDISTANCE 5.0  // maximum distance of person to camera
#define DISTANCEPOINT1 \
  1.0  // distance where close distance starts to decrease and far distance
       // starts to increase
#define DISTANCEPOINT2 \
  1.6  // distance where close distance ends and fardistance is at maximum
#define LEARNINGRATEINTEREST 0.3  // learning rate for final interest value

int handle_OpenCV_error(int status,
                        const char* func_name,
                        const char* err_msg,
                        const char* file_name,
                        int line,
                        void* userdata) {
  std::cerr << func_name << ": " << err_msg << std::endl;
  throw;
  return 0;
}

// Returns an string indicating the use of the program
string uso() {
  return "This program allows to track one or several people using plan view "
         "maps. You must indicate the following parameters.\n \t StereoVideo "
         "\n\t Num Frames Background: number of initial frames employed to "
         "create background \n\t Camera Height \n\t Camera tilt angle\n";
}

struct CallbackData {
  int waitKeyTime;
  int frame;
  StereoImage* StImage;
  PeopleTracker* Tracker;
  bool Finalizar, saveVideo;
  PlanViewMaps* PVMaps;
  IplImage* toVideo;
};

////////////////////////////////////
//
// Gets event from plan-view image
/////////////////////////////////////
void WindowEventsPlanView(int event, int x, int y, int flags, void* data)
// actions to perform when button is pressed in the window associated to the
// events function
{
  CallbackData* callback_data = (CallbackData*)data;
  /////////////////////////////
  // Adds people position to the list
  if (event == CV_EVENT_LBUTTONDOWN &&
      x >= callback_data->StImage->getWidth()) {
    x = x - callback_data->StImage->getWidth();
    float* height = (float*)(callback_data->PVMaps->hMap.hMap->imageData +
                             callback_data->PVMaps->hMap.hMap->widthStep * y);
    height += x;
    cout << "height=" << *height << endl;
    callback_data->Tracker->addPersonToTrack(pair<int, int>(x, y));
    cout << "Adding " << x << "," << y << endl;

    cvDestroyWindow("camera");
    /*askForGestureName();*/
  }

  if (event == CV_EVENT_MBUTTONDOWN) {
    if (callback_data->waitKeyTime == 0) {
      callback_data->waitKeyTime = 30;
    } else {
      callback_data->waitKeyTime = 0;
    }
  }

  if (event == CV_EVENT_MBUTTONDBLCLK) {
    callback_data->Finalizar = true;
    // AviWriter.close();
  }

  char buffer[150];
  if (event == CV_EVENT_RBUTTONDOWN) {
    sprintf(buffer, "/home/juant/images/image%d.bmp", callback_data->frame);
    cvSaveImage(buffer, callback_data->toVideo);
  }

  if (event == CV_EVENT_RBUTTONDBLCLK) {
    if (callback_data->saveVideo) {
      cout << "Not Saving Video" << endl;
      callback_data->saveVideo = false;
    } else {
      cout << "Saving Video" << endl;
      callback_data->saveVideo = true;
    }
  }
}

/////////////////////////////////////////////
// Transforming matrix from referential
//
/////////////////////////////////////////////
void createTransformMatrix(float PtuHeight,
                           float PtuPan,
                           float PtuTilt,
                           _3DTransformationMatrix& Matrix) throw(GUException) {
  float TMatrix[4][4];
  float tiltRadians = -(PtuTilt * M_PI) / 180.;
  float panRadians = -(PtuPan * M_PI) / 180.;
  float dax = 0.06;  // expressed in meters
  float day = 0.039116;  // expressed in meters
  float dby = 0.071628;
  float dcx = 0.0437134;
  float dcy = 0.04572;
  float dcz = 0.02159;

  TMatrix[0][0] = cos(panRadians);
  TMatrix[0][1] = sin(panRadians) * sin(tiltRadians);
  TMatrix[0][2] = sin(panRadians) * cos(tiltRadians);
  TMatrix[0][3] =
      (cos(panRadians) * dax + day * sin(panRadians) * sin(tiltRadians));

  TMatrix[1][0] = 0;
  TMatrix[1][1] = cos(tiltRadians);
  TMatrix[1][2] = -sin(tiltRadians);
  TMatrix[1][3] = (day * cos(tiltRadians) + dby + dcy);

  TMatrix[2][0] = -sin(panRadians);
  TMatrix[2][1] = cos(panRadians) * sin(tiltRadians);
  TMatrix[2][2] = cos(panRadians) * cos(tiltRadians);
  TMatrix[2][3] =
      -dax * sin(panRadians) + day * sin(tiltRadians) * cos(panRadians);

  TMatrix[3][0] = 0;
  TMatrix[3][1] = 0;
  TMatrix[3][2] = 0;
  TMatrix[3][3] = 1;

  TMatrix[1][3] += PtuHeight;
  Matrix.setMatrix(TMatrix);
}

/////////////////////////////////////////////
// MAIN PROGRAM
//
/////////////////////////////////////////////

int main(int argc, char** argv) {
  cv::redirectError(handle_OpenCV_error);
  // guavi::GUAviWrite AviWriter; //to save images to an avi file
  StereoSetVideoReader VideoReader;  // to read from the svs (video) file
  StereoImage* StImage;  // stereo image
  StereoGaussBackgroundCreator Background;  // keeps a background model
  gustereotools::gesturerecognition::PersonMaskExtractor PMaskExtractor;
  ImageComposer ImgComposer;
  StereoFiller8Lines StFiller;
  PlanViewMaps PVMaps;
  PeopleTracker Tracker;  // people tracker
  IplImage* rgbCameraImage, *toVideo, *hsvCameraImage, *auxGrey, *binForegorund,
      * colorMap = NULL;  // OpenCV images
  IplImage* imgFace, *halfImgFace, *mouthImg;
  float positionX[MAXIMUMPERSONS][FRAMESPERPERSON],
      positionZ[MAXIMUMPERSONS][FRAMESPERPERSON];
  int waitKeyTime = 0;  // waiting key time
  bool Finalizar = false, faceOpenCV;
  CvFont font;  // font if we want to write characters in the images
  boolean saveVideo = true;
  FuzzyMatlab closeFS, farFS;
  int headWidth, headHeight, goodHead;
  CvSeq* detectedFaces, *detectedEyes, *detectedMouth;
  CvRect* rectFace, *rectSmall, rectAux;
  int attentionClass, smileClass;
  SmileEstimator SmileEstimatorForFace;
  pair<float, float> personPos, mapPos, headPos;
  float* xyz, positionFaceX, positionFaceY, positionFaceZ;
  IplImage* binaryMask, *distanceMask, *colorMask, *smallFaceRGB,
      *smallFaceGrey;

  // IplImage *cameraBinaryMask;

  IplImage* oldBinaryMask[MAXIMUMPERSONS][FRAMESPERPERSON],
      *oldDistanceMask[MAXIMUMPERSONS][FRAMESPERPERSON],
      *oldColorMask[MAXIMUMPERSONS][FRAMESPERPERSON];
  bool analyseGestures[MAXIMUMPERSONS], firstTime[MAXIMUMPERSONS];
  float speed, RA, biggestDistance;
  int headx, heady, headlastx, firstheadxbinary, firstheadybinary,
      lastheadxbinary;
  bool headFound, tempBool;
  unsigned char* oldptrbinary[MAXIMUMPERSONS][FRAMESPERPERSON];
  long numberOfArmPixelsBinary, numberOfArmPixelsDistance,
      numberOfArmPixelsBinaryHistory, numberOfArmPixelsDistanceHistory;
  long numberDifferentBinaryPixels, numberDifferentRedPixels,
      numberDifferentBluePixels, numberDifferentGreenPixels,
      numberDifferentDistancePixels, numberOfPixels;
  vector<float> inRAP, inRAM, outRAP, outRAM, inCloseFS, outCloseFS, inFarFS,
      outFarFS;
  float attentionLevel[MAXIMUMPERSONS], movementArms[MAXIMUMPERSONS],
      smileLevel[MAXIMUMPERSONS], occludedPixelsRatio[MAXIMUMPERSONS],
      finalInterest[MAXIMUMPERSONS];
  FuzzyMatlab FMRAP, FMRAM;
  float* oldptrdistance[MAXIMUMPERSONS][FRAMESPERPERSON];
  int occludedPixels[MAXIMUMPERSONS];
  float pertenenciaLejos, dist, minDist;
  int numPixels, frame;

  // HaarObjectDetector
  // FaceDetector("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");
  // cv::CascadeClassifier FaceDetector1
  // ("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");

  try {
    // Check input
    if (argc != 5) {
      throw GUException(uso());
    }
    /// Create 3D transform matrix to move 3D points to a reference system in
    /// the floor
    _3DTransformationMatrix TMatrix;
    createTransformMatrix(atof(argv[3]), 0, atof(argv[4]), TMatrix);
    /// Open video and reads first stereo image to know the dimensions
    VideoReader.connect(argv[1]);
    StImage = VideoReader.produceNext()->get(
        0);  // retrieves the first stereo image (number 0)
    PMaskExtractor.setParams(StImage->getWidth(), StImage->getHeight(), 120,
                             120);
    /// allocateIplImages (OpenCV images)
    rgbCameraImage =
        cvCreateImage(cvSize(StImage->getWidth(), StImage->getHeight()), 8, 3);
    toVideo =
        cvCreateImage(cvSize(StImage->getWidth(), StImage->getHeight()), 8, 3);
    hsvCameraImage =
        cvCreateImage(cvSize(StImage->getWidth(), StImage->getHeight()), 8, 3);
    auxGrey =
        cvCreateImage(cvSize(StImage->getWidth(), StImage->getHeight()), 8, 1);
    binForegorund =
        cvCreateImage(cvSize(StImage->getWidth(), StImage->getHeight()), 8, 1);
    mouthImg = cvCreateImage(cvSize(25, 15), 8, 3);

    /// Write to files
    frame = 0;

    /// Attention Detection Variables
    closeFS.loadFromFile(
        "modelos/closefuzzy.fis");  // Sistema Difuso para "close"
    farFS.loadFromFile(
        "modelos/farfuzzy.fis");  // Sistema Difuso para "faraway"
    inRAP.resize(2);
    inRAM.resize(2);
    inCloseFS.resize(2);
    inFarFS.resize(3);
    FMRAP.loadFromFile("modelos/RAP.fis");  // Sistema difuso para brazos
    FMRAM.loadFromFile("modelos/RAM.fis");  // Sistema difuso para brazos
#define TODO
#ifdef TODO
    for (int i = 0; i < MAXIMUMPERSONS; i++) {
      attentionLevel[i] = 0;  // Empezar con todo a cero
      movementArms[i] = 0;
      smileLevel[i] = 0;
      occludedPixels[i] = 0;
      finalInterest[i] = 0;
      analyseGestures[i] = false;
      firstTime[i] = true;
      for (int j = 0; j < FRAMESPERPERSON; j++) {
        positionX[i][j] = 0;
        positionZ[i][j] = 0;
      }
    }

    /// Init background
    int nBackgrounFrames =
        atoi(argv[2]);  // number of frames employed for background modelling
    Background.setParams(StImage->getWidth(), StImage->getHeight());
    // read the first images and create the background
    for (int i = 0; i < nBackgrounFrames; i++) {
      // read next stereo image
      StImage = VideoReader.produceNext()->get(0);
      // gets the RGB color image
      StImage->colorImageToBGRBuffer(
          (unsigned char*)(rgbCameraImage->imageData), StereoImage::REFERENCE);
      // now, convert to HSV
      cvCvtColor(rgbCameraImage, hsvCameraImage, CV_BGR2HSV);
      // finally, create and update background.
      Background.getForeground(hsvCameraImage, StImage->getDepthImage(),
                               StImage->getValidityDepthMap());
      Background.update();
    }

    // create windows
    cvNamedWindow("foreground");
    cvNamedWindow("toVideoWindow");
    cvNamedWindow("composed");

    CallbackData data;
    data.waitKeyTime = waitKeyTime;
    data.Tracker = &Tracker;
    data.frame = frame;
    data.StImage = StImage;
    data.Finalizar = Finalizar;
    data.saveVideo = saveVideo;
    data.PVMaps = &PVMaps;
    data.toVideo = toVideo;

    cvSetMouseCallback("composed", WindowEventsPlanView,
                       &data);  // associates events to the window "rgbImage"

    cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, 0.4f,
               0.4f);  // parameters of the font which is used to write in the
                       // openCV images
    // Now, create plan view maps and show it waiting until
    // the user press on the person to be tracked
    cout << "Person autodetection OFF, Press over the person to track in the "
            "\"planview\" window with the left button" << endl;
    PVMaps.init(0.02, 8, 10, 5, 0, 2);
    Tracker.init(PVMaps);
    StFiller.setParams(StImage->getWidth(), StImage->getHeight(), 15, true);

    char cad[200];  // string
    while (!VideoReader.eof()) {  // cicle while the video does not end
      // read next stereo image
      StImage = VideoReader.produceNext()->get(0);
      // gets the RGB color image
      StImage->colorImageToBGRBuffer(
          (unsigned char*)(rgbCameraImage->imageData), StereoImage::REFERENCE);
      // now, convert to HSV
      cvCvtColor(rgbCameraImage, hsvCameraImage, CV_BGR2HSV);
      // finally, get foreground points
      IplImage* foreGround =
          Background.getForeground(hsvCameraImage, StImage->getDepthImage(),
                                   StImage->getValidityDepthMap(), -1, 1.5);
      // threshold the foreground image and erode to remove noise
      cvThreshold(foreGround, binForegorund, 225, 255, CV_THRESH_BINARY);
      cvErode(binForegorund, auxGrey);
      cvCopy(auxGrey, binForegorund);
      // Fill missing stereo information with the available one
      StFiller.fill(StImage, binForegorund);
      // Now, translate the points to the world coordinate center
      TMatrix.transform(StImage);
      // shows openCV images in the previously created windows
      PVMaps.create((unsigned char*)hsvCameraImage->imageData,
                    StImage->getDepthImage(), StImage->getWidth(),
                    StImage->getHeight(),
                    (unsigned char*)binForegorund->imageData);
      // now, show the image
      PVMaps.createDisplayableColorMap(
          (unsigned char*)rgbCameraImage->imageData, StImage->getDepthImage(),
          StImage->getWidth(), StImage->getHeight(), &colorMap, binForegorund);
      ImgComposer.createNormalComposedImage(PVMaps, rgbCameraImage,
                                            StImage->getDepthImage(),
                                            foreGround, &Tracker);

      // FaceDetector.detect(rgbCameraImage); // Detecta a la cara de todas las
      // personas en la imagen.
      // detectedFaces = FaceDetector.getFacePositions();
      toVideo =
          cvCloneImage(rgbCameraImage);  // Haz una copia clone de la imagen
      xyz = StImage->getDepthImage();  // Calcula la imagen de profundidad de la
                                       // imagen

      if (Tracker.getNumPeopleBeingTracked() == 0) {
        continue;
      }
        // Si hay personas a seguir...
        frame++;
        Tracker.iterate();  // Track iteration
        int indexWidth = 0;
        // Now, show every person mask
        for (int person = 0; person < Tracker.getNumPeopleBeingTracked();
             person++, indexWidth +=
                       2 * PMaskExtractor.getNormalizedColorMask()->width) {
          personPos = PVMaps.fromMapPosToWorldPos(
              Tracker.getMeanEstimatedPositions()[person]);
          PMaskExtractor.createMask(
              StImage, personPos, foreGround, 0.5,
              2.3);  // Mascara para detection movimiento brazos.

          // From DEA Work. Create and Update history of position and speed of
          // person
          binaryMask = PMaskExtractor.getBinaryMask();
          distanceMask = PMaskExtractor.getNormalizedMask();
          colorMask = PMaskExtractor.getColorMask();

          if (firstTime[person]) {  // Si es la primera vez que detectamos a
                                    // esta persona no hay historia de sus
                                    // posiciones... Decimos que todo es igual
                                    // para los frames de historia.
            for (int j = FRAMESPERPERSON - 1; j >= 0; j--) {
              positionX[person][j] = personPos.first;
              positionZ[person][j] = personPos.second;
              oldBinaryMask[person][j] = cvCloneImage(binaryMask);
              oldDistanceMask[person][j] = cvCloneImage(distanceMask);
              oldColorMask[person][j] = cvCloneImage(colorMask);
            }
          }
          for (int j = FRAMESPERPERSON - 1; j > 0; j--) {
            positionX[person][j] = positionX[person][j - 1];
            positionZ[person][j] = positionZ[person][j - 1];
          }
          positionX[person][0] = personPos.first;
          positionZ[person][0] = personPos.second;

          // Calculamos si la persona se ha movido ya que solo vamos a analisar
          // algunas cosas si la persona no se mueve mucho...
          speed = sqrt(pow(positionX[person][0] -
                               positionX[person][FRAMESPERPERSON - 1],
                           2) +
                       pow(positionZ[person][0] -
                               positionZ[person][FRAMESPERPERSON - 1],
                           2)) *
                  100.0;

          cvReleaseImage(&oldBinaryMask[person][FRAMESPERPERSON - 1]);
          cvReleaseImage(&oldDistanceMask[person][FRAMESPERPERSON - 1]);
          cvReleaseImage(&oldColorMask[person][FRAMESPERPERSON - 1]);
          for (int j = FRAMESPERPERSON - 1; j > 0; j--) {
            oldBinaryMask[person][j] = oldBinaryMask[person][j - 1];
            oldDistanceMask[person][j] = oldDistanceMask[person][j - 1];
            oldColorMask[person][j] = oldColorMask[person][j - 1];
          }
          oldBinaryMask[person][0] = cvCloneImage(binaryMask);
          oldDistanceMask[person][0] = cvCloneImage(distanceMask);
          oldColorMask[person][0] = cvCloneImage(colorMask);

          // cameraBinaryMask = cvCloneImage(StFiller.getComposedImage());

          // End of copy paste from DEA Work

          faceOpenCV = false;
          minDist = 100;
          goodHead = 0;
          for (int icv = 0; icv < (detectedFaces ? detectedFaces->total : 0);
               icv++) {  // Check if face can come from OpenCV detected Face
            rectFace = (CvRect*)cvGetSeqElem(detectedFaces, icv);
            cvRectangle(toVideo, cvPoint(rectFace->x, rectFace->y),
                        cvPoint(rectFace->x + rectFace->width,
                                rectFace->y + rectFace->height),
                        cvScalar(0, 0, 255), 2);  // vermelho
            positionFaceX = xyz[(rectFace->y + rectFace->height / 2) *
                                    StImage->getWidth() * 3 +
                                (rectFace->x + rectFace->width / 2) * 3];
            positionFaceY = xyz[(rectFace->y + rectFace->height / 2) *
                                    StImage->getWidth() * 3 +
                                (rectFace->x + rectFace->width / 2) * 3 + 1];

            positionFaceZ = xyz[(rectFace->y + rectFace->height / 2) *
                                    StImage->getWidth() * 3 +
                                (rectFace->x + rectFace->width / 2) * 3 + 2];

            dist = abs(positionFaceX - personPos.first);

            if (dist < ERRORDISTANCEMETERS &&
                positionFaceY >
                    1.3) {  // && abs(rectFace->width -
                            // (int)(96.0/personPos.second)) <
                            // 2.0/personPos.second*ERRORDISTANCEPIXELS && abs
                            // (rectFace->height -
                            // (int)(115.0/personPos.second)) <
                            // 4.0/personPos.second*ERRORDISTANCEPIXELS)
              faceOpenCV = true;  // Para recuperar precision en el tracking,
                                  // usamos cuando podemos la posicion de la
                                  // cara de openCV y buscamos a sus cordenadas
              if (dist < minDist) {
                minDist = dist;
                personPos.first = positionFaceX;
                personPos.second = positionFaceZ;
                positionZ[person][0] = personPos.second;
                headx = rectFace->x;
                heady = rectFace->y;
                headWidth = rectFace->width;
                headHeight = rectFace->height;
                goodHead = icv;
              }
            }
          }

          if (faceOpenCV) {
            rectFace = (CvRect*)cvGetSeqElem(detectedFaces,
                                             goodHead);  // Preparacion de la
                                                         // imagen de la cara
                                                         // para darle al
                                                         // detector de sonrisa
            rectAux = cvRect(rectFace->x, rectFace->y, rectFace->width,
                             rectFace->height);
            cvSetImageROI(rgbCameraImage, rectAux);
            smallFaceRGB = cvCreateImage(cvSize(40, 48), 8, 3);
            smallFaceGrey = cvCreateImage(cvSize(40, 48), 8, 1);
            cvResize(rgbCameraImage, smallFaceRGB);
            cvCvtColor(smallFaceRGB, smallFaceGrey, CV_BGR2GRAY);
            smileClass = SmileEstimatorForFace.estimate(
                smallFaceGrey);  // Detector de sonrisa
            cvResetImageROI(rgbCameraImage);

            switch (smileClass) {
              case 1:  // Low Smile
                smileLevel[person] =
                    smileLevel[person] *
                    (1 - LEARNINGRATESMILE);  // Version con "learning rate" por
                                              // lo de tener en cuenta la
                                              // historia. Si no nos interesa
                                              // hay que desactivar.
                break;
              case 2:  // High Smile
                smileLevel[person] =
                    smileLevel[person] * (1 - LEARNINGRATESMILE) +
                    LEARNINGRATESMILE;  // Version con "learning rate" por lo de
                                        // tener en cuenta la historia. Si no
                                        // nos interesa hay que desactivar.
                break;
            }
            if (smileClass == -1) {
              cout << "No smile detected for person " << person
                   << endl;  // No hemos detectado sonrisa
            }

            cvRectangle(toVideo, cvPoint(headx, heady),
                        cvPoint(headx + headWidth, heady + headHeight),
                        cvScalar(0, 255, 0), 3);  // verde cara do openCV
          }

          if (Tracker.getNumPeopleBeingTracked() > 1) {
            PMaskExtractor.isNearOthers(true);
          }

          if (!faceOpenCV) {  // Values come from tracker porque ninguna cara
                              // detectada por openCV parece corresponder a una
                              // persona que esta siendo seguida
            headPos.first = -1;
            if (PMaskExtractor.createMask(StImage, personPos, foreGround, 0.5,
                                          2.3)) {
              guopencv::Utilities::copyMakeBorder2(
                  PMaskExtractor.getNormalizedColorMask(),
                  ImgComposer.getLastComposed(), cvPoint(indexWidth, 0),
                  cvScalar(0, 0, 255), 1, 8, "Depth Mask", 0.4);
              guopencv::Utilities::copyMakeBorder2(
                  PMaskExtractor.getColorMask(), ImgComposer.getLastComposed(),
                  cvPoint(indexWidth +
                              PMaskExtractor.getNormalizedColorMask()->width,
                          0),
                  cvScalar(0, 0, 255), 1, 8, "Color Mask", 0.4);
              headPos = PMaskExtractor.getPersonHeadPosition();
              headWidth = (int)(96.0 / personPos.second);
              headHeight = (int)(115.0 / personPos.second);
              rectFace->x = headPos.first;
              rectFace->y = headPos.second;
              rectFace->width = headWidth;
              rectFace->height = headHeight;
              headPos.first += headWidth / 2;
              headPos.second += headHeight / 2;
            }
          }

          if (faceOpenCV) {
            attentionClass = 1;  // Como la atencion en esta version se basa en
                                 // el detector de caras frontales de openCV, si
                                 // hemos detectada una cara frontal entonces
                                 // persona esta atenta, sino consideramos que
                                 // la persona no esta atenta
          } else {
            attentionClass = 2;
            // Version con "learning rate" por lo de tener en cuenta la
            // historia. Si no nos interesa hay que desactivar.
            smileLevel[person] = smileLevel[person] * (1 - LEARNINGRATESMILE);
          }

          switch (attentionClass) {
            case 1:  // High Attention
              attentionLevel[person] =
                  attentionLevel[person] * (1 - LEARNINGRATEATTENTION) +
                  LEARNINGRATEATTENTION;  // Version con "learning rate" por lo
                                          // de tener en cuenta la historia. Si
                                          // no nos interesa hay que desactivar.
              break;
            case 2:  // Low Attention
              attentionLevel[person] =
                  attentionLevel[person] *
                  (1 - LEARNINGRATEATTENTION);  // Version con "learning rate"
                                                // por lo de tener en cuenta la
                                                // historia. Si no nos interesa
                                                // hay que desactivar.
              break;
          }

          cvNamedWindow("binary mask");
          cvShowImage("binary mask", binaryMask);

          if (analyseGestures[person]) {  // solo analisamos el movimiento de
                                          // los brazos si la persona no se esta
                                          // moviendo (o casi nada)
            cout << "IS NOT MOVING (A LOT) AND ATTENT" << endl;
            numberOfPixels = numberDifferentBinaryPixels =
                numberDifferentRedPixels = numberDifferentBluePixels =
                    numberDifferentGreenPixels = numberDifferentDistancePixels =
                        numberOfArmPixelsBinary = numberOfArmPixelsDistance =
                            numberOfArmPixelsBinaryHistory =
                                numberOfArmPixelsDistanceHistory = 0;
            for (int line = 0; line < binaryMask->height;
                 line++) {  // numero de lineas de la imagen...
              for (int x = 0; x < FRAMESPERPERSON;
                   x++) {  // numero de frames pasados que tomamos en cuenta (5
                           // en este caso)
                oldptrbinary[person][x] =
                    (unsigned char*)(oldBinaryMask[person][x]->imageData +
                                     oldBinaryMask[person][x]->widthStep *
                                         line);
                oldptrdistance[person][x] =
                    (float*)(oldDistanceMask[person][x]->imageData +
                             oldDistanceMask[person][x]->widthStep * line);
              }
              for (int column = 0; column < binaryMask->width;
                   column++) {  // numero de colonas de la imagne...
                // vamos a ver cuantos pixels que apartienen a un brazo(s) hay
                // en el entorno de la persona para el frame actual
                if (((int)(*oldptrbinary[person][0])) !=
                    0) {  // current binary image
                  numberOfPixels++;
                  if ((((float)((column - CENTERBINARYPICTUREX) *
                                (column - CENTERBINARYPICTUREX)) /
                        (float)(GESTUREMINIMUMX * GESTUREMINIMUMX)) +
                           ((float)((-line - CENTERBINARYPICTUREY) *
                                    (-line - CENTERBINARYPICTUREY)) /
                            (float)(GESTUREMINIMUMY * GESTUREMINIMUMY)) >
                       1.0) &&
                      (((float)((column - CENTERBINARYPICTUREX) *
                                (column - CENTERBINARYPICTUREX)) /
                        (float)(GESTUREMAXIMUMX * GESTUREMAXIMUMX)) +
                           ((float)((-line - CENTERBINARYPICTUREY) *
                                    (-line - CENTERBINARYPICTUREY)) /
                            (float)(GESTUREMAXIMUMY * GESTUREMAXIMUMY)) <
                       1.0) &&
                      line < 70) {
                    numberOfArmPixelsBinary++;
                  }
                }

                // vamos a ver cuantos pixels que apartienen a un brazo(s) hay
                // delante la persona para el frame actual (a una distancia
                // maxima como si los brazos se estuvieran moviendo delante la
                // persona)
                if ((float)(*oldptrdistance[person][0]) !=
                    0) {  // current distance image
                  if (((float)((column - CENTERBINARYPICTUREX) *
                               (column - CENTERBINARYPICTUREX)) /
                       (float)(GESTUREMAXIMUMX * GESTUREMAXIMUMX)) +
                          ((float)((-line - CENTERBINARYPICTUREY) *
                                   (-line - CENTERBINARYPICTUREY)) /
                           (float)(GESTUREMAXIMUMY * GESTUREMAXIMUMY)) <
                      1.0) {
                    if ((float)(*oldptrdistance[person][0]) >
                        UMBRALDISTANCEGESTURE) {
                      numberOfArmPixelsDistance++;
                    }
                  }
                }

                // vamos a ver cuantos pixels que apartienen a un brazo(s) hay
                // en el entorno de la persona para el total de los 5 frames en
                // memoria
                tempBool = false;
                for (int i = 0; i < FRAMESPERPERSON; i++)  // numero de frames
                                                           // pasados que
                                                           // tomamos en cuenta
                                                           // (5 en este caso)
                  if (((int)(*oldptrbinary[person][i])) != 0) {
                    tempBool = true;
                  }
                if (tempBool) {
                  if ((((float)((column - CENTERBINARYPICTUREX) *
                                (column - CENTERBINARYPICTUREX)) /
                        (float)(GESTUREMINIMUMX * GESTUREMINIMUMX)) +
                           ((float)((-line - CENTERBINARYPICTUREY) *
                                    (-line - CENTERBINARYPICTUREY)) /
                            (float)(GESTUREMINIMUMY * GESTUREMINIMUMY)) >
                       1.0) &&
                      (((float)((column - CENTERBINARYPICTUREX) *
                                (column - CENTERBINARYPICTUREX)) /
                        (float)(GESTUREMAXIMUMX * GESTUREMAXIMUMX)) +
                           ((float)((-line - CENTERBINARYPICTUREY) *
                                    (-line - CENTERBINARYPICTUREY)) /
                            (float)(GESTUREMAXIMUMY * GESTUREMAXIMUMY)) <
                       1.0) &&
                      line < 70) {
                    numberOfArmPixelsBinaryHistory++;
                  }
                }

                // vamos a ver cuantos pixels que apartienen a un brazo(s) hay
                // delante la persona para el total de los 5 frames en memoria
                // (a una distancia maxima como si los brazos se estuvieran
                // moviendo delante la persona)
                tempBool = false;
                for (int i = 0; i < FRAMESPERPERSON; i++) {
                  if (((float)(*oldptrdistance[person][i])) != 0) {
                    tempBool = true;
                  }
                }
                if (tempBool) {
                  if (((float)((column - CENTERBINARYPICTUREX) *
                               (column - CENTERBINARYPICTUREX)) /
                       (float)(GESTUREMAXIMUMX * GESTUREMAXIMUMX)) +
                          ((float)((-line - CENTERBINARYPICTUREY) *
                                   (-line - CENTERBINARYPICTUREY)) /
                           (float)(GESTUREMAXIMUMY * GESTUREMAXIMUMY)) <
                      1.0) {
                    biggestDistance = 0;
                    for (int i = 0; i < FRAMESPERPERSON; i++)
                      if (((float)(*oldptrdistance[person][i])) >
                          biggestDistance) {
                        biggestDistance = *oldptrdistance[person][i];
                      }
                    if (biggestDistance > UMBRALDISTANCEGESTURE) {
                      numberOfArmPixelsDistanceHistory++;
                    }
                  }
                }

                for (int x = 0; x < FRAMESPERPERSON; x++) {
                  oldptrbinary[person][x]++;
                  oldptrdistance[person][x]++;
                }
              }
            }

            // 200 y 500 son umbrales que hemos calculado para brazo(s)
            // parado(s) y brazo(s) moviendose
            cout << "TOTAL PIXELS: " << numberOfPixels
                 << " ARMS SIDE: " << numberOfArmPixelsBinary
                 << " ARMS FRONT: " << numberOfArmPixelsDistance
                 << " ARMS SIDE MOVE: " << numberOfArmPixelsBinaryHistory
                 << " ARMS FRONT MOVE: " << numberOfArmPixelsDistanceHistory
                 << endl;
            if (numberOfArmPixelsBinary > 200) {
              numberOfArmPixelsBinary = 200;
            }
            if (numberOfArmPixelsDistance > 200) {
              numberOfArmPixelsDistance = 200;
            }
            if (numberOfArmPixelsBinaryHistory > 500) {
              numberOfArmPixelsBinaryHistory = 500;
            }
            if (numberOfArmPixelsDistanceHistory > 500) {
              numberOfArmPixelsDistanceHistory = 500;
            }
          }

          if (speed < SPEED) {
            analyseGestures[person] =
                true;  // si velocidad inferior a un limite entonces analisamos
                       // los gestos, sino no analisamos (porque quizas no se
                       // trata de un movimiento de brazos pero de todo el
                       // cuerpo)
          } else {
            analyseGestures[person] = false;
          }

          if (analyseGestures[person]) {  // si es para analisar...
            try {  // preparamos los vectores de entrada del sistema difuso de
                   // analise de gestos
              inRAP[0] = numberOfArmPixelsBinary;
              inRAP[1] = numberOfArmPixelsDistance;
              inRAM[0] = numberOfArmPixelsBinaryHistory;
              inRAM[1] = numberOfArmPixelsDistanceHistory;
              FMRAP.process(inRAP, outRAP);  // parado
              FMRAM.process(inRAM, outRAM);  // movimiento
            } catch (std::exception& ex) {
              cout << ex.what() << endl;
            }

            if (outRAP[0] > outRAM[0]) {
              RA = outRAP[0];
            } else {
              RA = outRAM[0];
            }

            movementArms[person] =
                movementArms[person] * (1 - LEARNINGRATEARMS) +
                LEARNINGRATEARMS * RA;  // Version con "learning rate" por lo de
                                        // tener en cuenta la historia. Si no
                                        // nos interesa hay que desactivar.

            // debugging
            if (movementArms[person] <= 0.2) {
              sprintf(cad, "RA: V. Little (%.2f)", movementArms[person]);
            }
            if (movementArms[person] > 0.2 && movementArms[person] <= 0.4) {
              sprintf(cad, "RA: Little (%.2f)", movementArms[person]);
            }
            if (movementArms[person] > 0.4 && movementArms[person] <= 0.6) {
              sprintf(cad, "RA: Medium (%.2f)", movementArms[person]);
            }
            if (movementArms[person] > 0.6 && movementArms[person] <= 0.8) {
              sprintf(cad, "RA: Much (%.2f)", movementArms[person]);
            }
            if (movementArms[person] > 0.8) {
              sprintf(cad, "RA: V. Much (%.2f)", movementArms[person]);
            }
          }
        }

        // Analyse occlusion: Compares person[i] to every other person if
        // distance of the other person to the camera is smaller than
        // person[i]'s distance. Counts those pixels (which occlude person[i])
        for (int i = 0; i < Tracker.getNumPeopleBeingTracked(); i++) {
          numPixels = 1;
          occludedPixels[i] = 0;
          for (int line = 0; line < binaryMask->height; line++) {
            for (int j = 0; j < Tracker.getNumPeopleBeingTracked(); j++) {
              oldptrbinary[j][0] =
                  (unsigned char*)(oldBinaryMask[j][0]->imageData +
                                   oldBinaryMask[j][0]->widthStep * line);
            }

            for (int column = 0; column < binaryMask->width; column++) {
              if (((int)(*oldptrbinary[i][0])) != 0) {
                numPixels++;
              }
              for (int j = 0; j < Tracker.getNumPeopleBeingTracked(); j++) {
                if (i != j && positionZ[j][0] < positionZ[i][0]) {
                  if (((int)(*oldptrbinary[i][0])) != 0 &&
                      ((int)(*oldptrbinary[j][0])) != 0) {
                    occludedPixels[i]++;
                  }
                }
              }
              for (int j = 0; j < Tracker.getNumPeopleBeingTracked(); j++) {
                oldptrbinary[j][0]++;
              }
            }
          }
          occludedPixelsRatio[i] = (float)occludedPixels[i] / (float)numPixels;
        }

        // Compute person's interest. Uses the output of either a "far fuzzy
        // system" or "close fuzzy system"" according to the distance of the
        // current person.
        for (int i = 0; i < Tracker.getNumPeopleBeingTracked(); i++) {
          cout << "Distance person " << i << " is " << positionZ[i][0]
               << " meters. Activated ";
          if (positionZ[i][0] > MINDISTANCE &&
              positionZ[i][0] <
                  DISTANCEPOINT1) {  // La persona esta cerca de la camera...
            inCloseFS[0] = attentionLevel[i];  // entrada atencion para FS Close
            inCloseFS[1] = smileLevel[i];      // entrada sonrisa para FS Close
            closeFS.process(inCloseFS, outCloseFS);
            // finalInterest[i] = finalInterest[i]*(1-LEARNINGRATEINTEREST) +
            // LEARNINGRATEINTEREST*outCloseFS[0]; // Version con "learning
            // rate" por lo de tener en cuenta la historia. Si no nos interesa
            // hay que desactivar.
            finalInterest[i] =
                outCloseFS[0];  // salida Interest = salida FS Close
          } else if (positionZ[i][0] > DISTANCEPOINT2 &&
                     positionZ[i][0] < MAXDISTANCE) {  // La persona esta lejos
                                                       // de la camera...
            inFarFS[0] = attentionLevel[i];  // entrada atencion para FS Faraway
            inFarFS[1] =
                movementArms[i];  // entrada movimiento brazos para FS Faraway
            inFarFS[2] = occludedPixelsRatio[i];  // entrada pixels occlusion
                                                  // para FS Faraway
            farFS.process(inFarFS, outFarFS);
            // finalInterest[i] = finalInterest[i]*(1-LEARNINGRATEINTEREST) +
            // LEARNINGRATEINTEREST*outFarFS[0]; // Version con "learning rate"
            // por lo de tener en cuenta la historia. Si no nos interesa hay que
            // desactivar.
            finalInterest[i] =
                outFarFS[0];  // salida Interest = salida FS Faraway
          } else if (positionZ[i][0] > DISTANCEPOINT1 &&
                     positionZ[i][0] <
                         DISTANCEPOINT2) {  // La persona esta entre cerca y
                                            // lejos... utilisamos los 2
                                            // Sistemas Difusos...
            inCloseFS[0] = attentionLevel[i];
            inCloseFS[1] = smileLevel[i];
            closeFS.process(inCloseFS, outCloseFS);
            inFarFS[0] = attentionLevel[i];
            inFarFS[1] = movementArms[i];
            inFarFS[2] = occludedPixelsRatio[i];
            farFS.process(inFarFS, outFarFS);
            pertenenciaLejos = (positionZ[i][0] - DISTANCEPOINT1) /
                               (DISTANCEPOINT2 - DISTANCEPOINT1);
            // finalInterest[i] =
            // finalInterest[i]*(1-LEARNINGRATEINTEREST)+LEARNINGRATEINTEREST*((1-pertenenciaLejos)*outCloseFS[0]
            // + pertenenciaLejos*outFarFS[0]); // Version con "learning rate"
            // por lo de tener en cuenta la historia. Si no nos interesa hay que
            // desactivar.
            finalInterest[i] =
                (1 - pertenenciaLejos) * outCloseFS[0] +
                pertenenciaLejos * outFarFS[0];  // para distancias intermedias
                                                 // el peso de cada FS depende
                                                 // de la distancia a que la
                                                 // persona se encuentra
          } else {
            finalInterest[i] = finalInterest[i];
            cout << "ANY. Person OUTSIDE camera distance limits." << endl;
          }
        }


      cvShowImage("composed", ImgComposer.getLastComposed());
      cvShowImage("foreground", foreGround);
      cvShowImage("toVideoWindow", toVideo);

      // Save to avi
      /*  if (!AviWriter.isValid())
          { //open file if not opened
              AviWriter.setParams (toVideo->width,toVideo->height, 15);
              AviWriter.open("video.avi");
          }
          if (saveVideo) AviWriter.addFrame((unsigned
         char*)toVideo->imageData,false);
      */
      cvReleaseImage(&imgFace);
      cvReleaseImage(&halfImgFace);

      // waits for the reader to press a key
      cvWaitKey(waitKeyTime);
    }
#endif
    // closes the avi file.
    // AviWriter.close();
  }

  catch (std::exception& ex) {
    // AviWriter.close();
    cout << ex.what() << endl;
  }
}
