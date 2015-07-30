/***************************************************************************
                          guaviwrite.cpp  -  description
                             -------------------
    begin                : mi�ene 14 2004
    copyright            : (C) 2004 by
    email                :
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <guavi/revel.h>
#include <iostream>
#include <guavi/guaviwrite.h>
#include <string.h>
using namespace std;
namespace guavi{
//////////////////////////////////
//
//
//////////////////////////////////
GUAviWrite::GUAviWrite()
{
  width=height=-1;
  fps=0.;
  frame.width =frame.height = -1;
  frame.pixels =NULL;
 _isValid=false;
}
//////////////////////////////////
//
//
//////////////////////////////////
GUAviWrite::~GUAviWrite()
{
  close();
}

//////////////////////////////////
//
//
//////////////////////////////////
void GUAviWrite::setParams(int Width,int Height,double Fps,int Quality)throw(GUException)
{
if (Width<=0 || Height<=0 || !(0<=Fps && Fps<=120))
  throw GUException("GUAviWrite::setParams incorrect params");
  width=Width;
  height=Height;
  fps=Fps;
  nPixels= width*height;
}

//////////////////////////////////
//
//
//////////////////////////////////
void GUAviWrite::open(const string Path) throw(GUException)
{
  if (REVEL_API_VERSION != Revel_GetApiVersion())
    throw GUException("GUAviWrite::open Revel library incorrect API");

  if (isValid()) close();
//    throw GUException("GUAviWrite::open it's necessary to close the file before opening a new one");
  if (width==-1)
    throw GUException("GUAviWrite::open setParams has not been called yet");

    revError = Revel_CreateEncoder(&encoderHandle);
    if (revError != REVEL_ERR_NONE)
	throw GUException("GUAviWrite::open  Revel Error while creating encoder");

   ///Initialize
   Revel_InitializeParams(&revParams);
   revParams.width = width;
   revParams.height = height;
   revParams.frameRate = fps;
   revParams.hasAudio = 0;

   revError = Revel_EncodeStart(encoderHandle, Path.c_str(), &revParams);
   if (revError != REVEL_ERR_NONE)
	throw GUException("GUAviWrite::open Revel Error while starting encoding");

    iCurrentFrameIndex=0;
    ///Set frame parameters
    frame.width = width;
    frame.height = height;
    frame.bytesPerPixel = 3;
    frame.pixelFormat = REVEL_PF_BGR;
    if (frame.pixels !=NULL ) delete [] (unsigned char*)frame.pixels ;
    frame.pixels = new unsigned char[width*height*3];
    pixelPtr=(unsigned char*)frame.pixels ;
    _isValid=true;
}

//////////////////////////////////
//
//
//////////////////////////////////
void GUAviWrite::addFrame(unsigned char *data,bool swapBGR)throw(GUException)
{
int frameSize;

   if (!isValid())
      throw GUException(" GUAviWrite::addFrame not a valid object");

   ///Swap data if required
   if (swapBGR){
    //swap rgb to bgr
    for(int i=0;i<nPixels*3;i+=3){
      pixelPtr[i]=data[i+2];//b
      pixelPtr[i+1]=data[i+1]; //g
      pixelPtr[i+2]=data[i]; //r
    }
   }
  else memcpy(frame.pixels,data,getWidth()*getHeight()*3);
  ///Encode
  revError = Revel_EncodeFrame(encoderHandle, &frame, &frameSize);
  if (revError != REVEL_ERR_NONE)
    throw GUException("GUAviWrite::addFrame Revel Error while writing frame");

    iCurrentFrameIndex++;
}

//////////////////////////////////
//
//
//////////////////////////////////
void GUAviWrite::addFrame(unsigned char *r,unsigned char *g,unsigned char *b)throw(GUException)
{
    if (!isValid())
      throw GUException(" GUAviWrite::addFrame not a valid object");
    //pasamos la informacion a la imagen para agragarla
    int counter=0;
    for(int i=0;i<nPixels*3;i+=3,counter++)
    {
      pixelPtr[i]=b[counter];
      pixelPtr[i+1]=g[counter];
      pixelPtr[i+2]=r[counter];
    }

  ///Encode
  int frameSize;
  revError = Revel_EncodeFrame(encoderHandle, &frame, &frameSize);
  if (revError != REVEL_ERR_NONE)
    throw GUException("GUAviWrite::addFrame Revel Error while writing frame");

    iCurrentFrameIndex++;
}


//////////////////////////////////
//
//
//////////////////////////////////
void GUAviWrite::close()throw(GUException)
{
  if (!isValid())
    return;
  if (frame.pixels !=NULL ) delete  [](unsigned char*) frame.pixels ;
  int totalSize;
  revError = Revel_EncodeEnd(encoderHandle, &totalSize);
  if (revError != REVEL_ERR_NONE)
      throw GUException("GUAviWrite::close Revel Error while ending encoding");
  Revel_DestroyEncoder(encoderHandle);
  width=height=-1;
  _isValid=false;
}

//////////////////////////////////
//
//
//////////////////////////////////
const unsigned char *  GUAviWrite::currentFrame()throw(GUException)
{
  if (!isValid())
   throw GUException(" GUAviWrite::currentFrame not a valid object");

 return pixelPtr;
}
};
