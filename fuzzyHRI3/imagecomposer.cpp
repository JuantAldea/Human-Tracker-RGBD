/***************************************************************************
 *   Copyright (C) 2006 by salinas   *
 *   salinas@localhost   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include "imagecomposer.h"


////////////////////////////////////
//
//
////////////////////////////////////
ImageComposer::ImageComposer()
{
    Composed = colorMap = NULL;
    cvInitFont(& font, CV_FONT_HERSHEY_DUPLEX, 0.55f, 0.4f, 0, 1, CV_AA);
    cvInitFont(& fontText, CV_FONT_HERSHEY_DUPLEX, 0.5f, 0.5f);

    _particleImage = NULL;
}



////////////////////////////////////
//
//
////////////////////////////////////
ImageComposer::~ImageComposer()
{
}


////////////////////////////////////
//
//
////////////////////////////////////
void ImageComposer::createNormalComposedImage(PlanViewMaps &Maps, IplImage *cameraImage,
        float*XYZ, IplImage *mask, PeopleTracker *PeopleT)
{
///Damos memoria a  la imagen de salida si no esta creada o si ha cambiado los tamaños empleados
    int width = cameraImage->width + Maps.nCellsX;
    int height = max(cameraImage->height, Maps.nCellsY);
    if (Composed != NULL) {
        if (Composed->width != width || Composed->height != height) {
            cvReleaseImage(&Composed);
            Composed = cvCreateImage(cvSize(width, height), 8, 3);
        }
    } else {
        Composed = cvCreateImage(cvSize(width, height), 8, 3);
    }
    cvSetZero(Composed);

///Copiamos la imagen de referencia
//Ahora la pasamos a la compuesta
    int offset = (Composed->height - cameraImage->height) / 2;
    cvSetImageROI(Composed, cvRect(0, offset, cameraImage->width, cameraImage->height));
    cvCopy(cameraImage, Composed);
    cvResetImageROI(Composed);

///Creamos el mapa de color

    Maps.createDisplayableColorMap((unsigned char*)cameraImage->imageData, XYZ, cameraImage->width,
                                   cameraImage->height, &colorMap, mask);

    int offsetColorMap = (Composed->height - colorMap->height) / 2;



///Ponemos la imagen de particulas si se ha indicado
    if (_particleImage) {
        //Primero, creamos _auxPar... que es el mapa con las particulas
        for (int y = 0; y < _particleImage->height; y++) {
            unsigned char *particleImg = (unsigned char *)_particleImage->imageData +
                                         _particleImage->widthStep * y;
            unsigned char *colorMapImg = (unsigned char *)colorMap->imageData + colorMap->widthStep * y;
            for (int x = 0; x < colorMap->width; x++) {
                if (particleImg[0] != 0 || particleImg[1] != 0 || particleImg[2] != 0) {
                    *colorMapImg++ = *particleImg++;
                    *colorMapImg++ = *particleImg++;
                    *colorMapImg++ = *particleImg++;
                } else {
                    colorMapImg += 3;
                    particleImg += 3;
                }
            }
        }
    }

//Pintamos sobre el los cuadrados de las posiciones
    if (PeopleT != 0) {
        list< pair<PersonTracker*, int> >::iterator itPerson = PeopleT->Trackers.begin();
        for (; itPerson != PeopleT->Trackers.end(); itPerson++) {
            drawPersonInformation((*itPerson), PeopleT->getHalfPersonCells());
        }
    }

//Copiamos _auxParticleImage en la imagen compuesta
///Copiamos mapa de color a la derecha de la imagen de referencia
    cvSetImageROI(Composed, cvRect(cameraImage->width, offsetColorMap, colorMap->width,
                                   colorMap->height));
    cvCopy(colorMap, Composed);
    cvResetImageROI(Composed);
    cvRectangle(Composed, cvPoint(cameraImage->width, offsetColorMap),
                cvPoint(cameraImage->width + colorMap->width - 1, offsetColorMap + colorMap->height - 1),
                cvScalar(0, 0, 0), 1, CV_AA);

//Ahora agregamos texto si se pidio
    if (addedText != "") {
        cvPutText(Composed, addedText.c_str(), cvPoint(20, 20), &fontText, cvScalar(0, 0, 0));
        addedText = ""; //se borra para la proxima
    }

}

////////////////////////////////////
//
//
////////////////////////////////////
void ImageComposer::addText(string text)
{
    addedText = text;
}


////////////////////////////////////
//
//
////////////////////////////////////
IplImage *ImageComposer::getLastComposed()
{
    return Composed;
}

////////////////////////////////////
//
//
////////////////////////////////////
void ImageComposer::drawPersonInformation(pair<PersonTracker*, int> &Person,
        int halfPersonSize)
{
    list<pair<int, int> >::iterator positions = Person.first->getPositionHistory().begin();
    int sizePosition = halfPersonSize / 5;
// void cvCircle( CvArr* img, CvPoint center, int radius, CvScalar color,
//                int thickness=1, int line_type=8, int shift=0 );


    for (; positions != Person.first->getPositionHistory().end(); positions++) {
        cvCircle(colorMap, cvPoint(positions->first, positions->second), sizePosition,
                 Person.first->particleColor, 1, CV_AA);
    }
    PersonTracker::PersonStatus PS;
    Person.first->getPersonStatus(PS);
    if (PS.state == PersonTracker::Detected) {
        cvRectangle(colorMap, cvPoint(PS.position.first - halfPersonSize,
                                      PS.position.second - halfPersonSize), cvPoint(PS.position.first + halfPersonSize,
                                              PS.position.second + halfPersonSize), Person.first->particleColor, 1, CV_AA);
    } else {
        cvCircle(colorMap, cvPoint(PS.position.first, PS.position.second), halfPersonSize,
                 Person.first->particleColor, 1, CV_AA);
    }
//Ahora dibujamos la posicion actual en grande
    char cad[100];
    sprintf(cad, "%d", Person.second);
    cvPutText(colorMap, cad, cvPoint(PS.position.first + halfPersonSize,
                                     PS.position.second + halfPersonSize), &font, cvScalar(0, 20, 200));


}


