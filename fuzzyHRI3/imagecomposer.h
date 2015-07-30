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
#ifndef IMAGECOMPOSER_H
#define IMAGECOMPOSER_H
#include <gustereotools/planviewmaps.h>
#include <opencv/cv.h>
#include <guia/condensation.h>
#include <gustereotools/peopletracker.h>
using namespace condensation;
using namespace gustereotools::planviewmaps;
using namespace gustereotools::pv_peopletrackers;
/**
    @author salinas <salinas@localhost>
*/
class ImageComposer
{
public:
    /**
     */
    ImageComposer();
    /**Createsa normal composed image
    *@param maps from the stereo information
    */
    void createNormalComposedImage(PlanViewMaps &Maps, IplImage *cameraImage, float*XYZ,
                                   IplImage *mask, PeopleTracker *PeopleT = NULL);

    /**Createsa normal composed image
    *@param maps from the stereo information
    *@param Camera stereo camera
    *@param positions people positions if avaiable, the positions are printed as squares
    *@param halfSquareSize half size of the squares used to represent the people posisitions
    */
//    void createNormalComposedImage(PlanViewMaps &Maps,IplImage *cameraImage,float*XYZ,IplImage *mask,vector<pair<int,int> > *positions=NULL,int halfSquareSize=10);

    /**Call this function to superimpose in the composed image the particles employed.
     *@param image with the particles. This is a pointer to the image that is updates at each iteration
     */
    void showParticleImage(IplImage *particleImage)
    {
        _particleImage = particleImage;
    }
    /**Adds in the next image the text passed at the top of it
     */
    void addText(string text);
    /**Call this function to stop superimposing in the particles image.
     */
    void hideParticleImage()
    {
        _particleImage = NULL;
    }

    /**Returns the last image composed
     */
    IplImage *getLastComposed();
    /**
     */
    ~ImageComposer();
private:

    void drawPersonInformation(pair<PersonTracker*, int> &p, int halfPersonSize);
    IplImage *colorMap, *Composed;
    CvFont font, fontText;
    IplImage *_particleImage;
    string addedText;

};

#endif
