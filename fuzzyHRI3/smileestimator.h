/***************************************************************************
 *   Copyright (C) 2006 by Rui Paúl   *
 *   ruipaul@decsai.ugr.es   *
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

#ifndef SMILEESTIMATOR_H
#define SMILEESTIMATOR_H

//#include <gustereo2/stcamera.h>
#include <opencv/cv.h>
#include <vector>
#include <guia/svm.h>
#include <guia/pca.h>

//using namespace gustereo2;
using namespace std;
using namespace guia_addon;

/**\brief  This class performs an estimation of the person smile by using SVM previously trained
@author Rui Paúl
*/
class SmileEstimator
{

public:

    /**
    */
    SmileEstimator();

    /**
    */
    ~SmileEstimator();

    /** Starts
    */
    void init();

    /** Estimate the current person smile.
    * You must pass the image of the mouth to estimate the smile.
    *@param imgFace image of the face
    */
    int estimate(IplImage *imgMouth);


private:

    IplImage *imageOfFace;
    Svm trainedSVM;
    guia_addon::PCA pca;


};

#endif
