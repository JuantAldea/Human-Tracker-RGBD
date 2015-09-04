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
#include "smileestimator.h"
#include <iostream>
#include <opencv/highgui.h>
#include <guopencv/lineariterator.h>
#include <guia/pca.h>
#include <guia/patternset.h>
#include <guvision/gucolor.h>
#include <guia/svm.h>
#include <gu/gustringutils.h>
#include <gu/gurandom.h>
using namespace std;
using namespace guopencv::iterators;
using namespace guia;
using namespace guia_addon;

////////////////////////////
//
//
////////////////////////////
SmileEstimator::SmileEstimator()
{
    pca.readFromFile("modelos/allDBsPGM.pca");
    trainedSVM.loadModelFromFile("modelos/allDBsPGM.svm.model");
    trainedSVM.loadScaleFile("modelos/allDBsPGM.svm.range");
}

////////////////////////////
//
//
////////////////////////////
SmileEstimator::~SmileEstimator()
{
}

////////////////////////////
//
//
////////////////////////////
void SmileEstimator::init()
{

}

////////////////////////////
//
//
////////////////////////////
int SmileEstimator::estimate(IplImage *imgFace)
{
    int i;
    char nameFile[50];
    //int correct = 0;
    int smileClass;


    //cvNamedWindow("image2",CV_WINDOW_AUTOSIZE);
    imageOfFace = imgFace;

    Pattern<double, double> PS, PSPCA;
    PS.getData().resize(40 * 48);
    PSPCA.getData().resize(40 * 48);
    LinearIterator<unsigned char>Li;
    i = 0;
    for (Li.start(imageOfFace); !Li.eof(); Li.next()) {
        PS.getData()[i++] = *Li.get();
    }

    pca.fromOriginalToTransformed(PS, PSPCA, 100);

    smileClass = trainedSVM.evaluate(PSPCA.getData());
    return (smileClass);
}
