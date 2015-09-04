/***************************************************************************
                          gucolor.cpp  -  description
                             -------------------
    begin                : mié jul 28 2004
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

#include <guvision/gucolor.h>
#include <cmath>
using namespace std;
////////////////////////////////////
//
//
////////////////////////////////////
GUColor::GUColor()
{
  components.resize(3);
  components[0]=-1;
  components[1]=-1;
  components[2]=-1;  
  type="NoType";
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor::GUColor(const GUColor &GC)
{
  components=GC.components;
  type=GC.type;
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor::GUColor(double c1,double c2,double c3,string _type)throw (string){
    components.resize(3);
    components[0]=c1;
    components[1]=c2;
    components[2]=c3;
    type="NoType";
    if (!isValidType(_type))   throw string("Incorrect color type");
    type=_type;
}

////////////////////////////////////
//
//
////////////////////////////////////
void GUColor::setParams(double c1,double c2,double c3,string _type)throw (string)
{
    components.resize(3);
    components[0]=c1;
    components[1]=c2;
    components[2]=c3;
    type="NoType";
    if (!isValidType(_type)) throw string("GUColor::setParams Incorrect color type");
    type=_type;
}
////////////////////////////////////
//
//
////////////////////////////////////
GUColor::~GUColor()
{

}

////////////////////////////////////
//
//
////////////////////////////////////
double & GUColor::operator[](unsigned int index) throw(string) {
    if (index>2) throw string("GUColor::operator[] incorrect index");
    return components[index];
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor & GUColor::operator=(const GUColor&C) throw(string)
{
components=C.components;
type=C.type;
return (*this);
}

////////////////////////////////////
//
//
////////////////////////////////////
ostream &operator<<(ostream &str,GUColor &c)
{
  str<< c.components[0]<<" "<<c.components[1]<<" "<<c.components[2]<<endl;
  return str;
}

////////////////////////////////////
//
//
////////////////////////////////////
string GUColor::c_str()
{
string str;
char cad[100];
sprintf(cad,"%lf",components[0]);
str+="("+string(cad)+",";
sprintf(cad,"%lf",components[1]);
str+=string(cad)+",";
sprintf(cad,"%lf",components[2]);
str+=string(cad)+")";
return str;
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toRGB()   throw(string)
{
  if (type=="RGB") return *this;
  if (type=="NormalizedRGB")
     throw string(" GUColor::toRGB color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return HSV2RGB(*this);
  if (type=="YCbCr") return YCbCr2RGB(*this);
  if (type=="YUV") return YUV2RGB(*this);
  if (type=="YIQ") return YIQ2RGB(*this);
  if (type=="XYZD65") return XYZD652RGB(*this);
  if (type=="Lab") return Lab2RGB(*this);
  if (type=="Luv") return Luv2RGB(*this);
  if (type=="CMY") return CMY2RGB(*this);
  if (type=="TSL") return TSL2RGB(*this);
  if (type=="CIExy") return CIExy2RGB(*this);
  throw string(" GUColor::toRGB incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toNormalizedRGB()  throw(string)
{
  if (type=="RGB") return RGB2NormalizedRGB(*this);
  if (type=="NormalizedRGB") return *this;
  if (type=="HSV") return RGB2NormalizedRGB(HSV2RGB(*this));
  if (type=="YCbCr") return RGB2NormalizedRGB(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2NormalizedRGB(YUV2RGB(*this));
  if (type=="YIQ") return RGB2NormalizedRGB(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2NormalizedRGB(XYZD652RGB(*this));
  if (type=="Lab") return RGB2NormalizedRGB(Lab2RGB(*this));
  if (type=="Luv") return RGB2NormalizedRGB(Luv2RGB(*this));
  if (type=="CMY") return RGB2NormalizedRGB(CMY2RGB(*this));
  if (type=="TSL") return RGB2NormalizedRGB(TSL2RGB(*this));
  if (type=="CIExy") return RGB2NormalizedRGB(CIExy2RGB(*this));
  throw string(" GUColor::toNormalizedRGB incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toHSV()  throw(string)
{
  if (type=="RGB") return RGB2HSV(*this);
  if (type=="NormalizedRGB")
    throw string(" GUColor::toHSV color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return *this;
  if (type=="YCbCr") return RGB2HSV(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2HSV(YUV2RGB(*this));
  if (type=="YIQ") return RGB2HSV(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2HSV(XYZD652RGB(*this));
  if (type=="Lab") return RGB2HSV(Lab2RGB(*this));
  if (type=="Luv") return RGB2HSV(Luv2RGB(*this));
  if (type=="CMY") return RGB2HSV(CMY2RGB(*this));
  if (type=="TSL") return RGB2HSV(TSL2RGB(*this));
  if (type=="CIExy") return RGB2HSV(CIExy2RGB(*this));
  throw string(" GUColor::toHSV incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toYCbCr() throw(string)
{
  if (type=="RGB") return RGB2YCbCr(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toYCbCr color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2YCbCr(HSV2RGB(*this));
  if (type=="YCbCr") return *this;
  if (type=="YUV") return RGB2YCbCr(YUV2RGB(*this));
  if (type=="YIQ") return RGB2YCbCr(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2YCbCr(XYZD652RGB(*this));
  if (type=="Lab") return RGB2YCbCr(Lab2RGB(*this));
  if (type=="Luv") return RGB2YCbCr(Luv2RGB(*this));
  if (type=="CMY") return RGB2YCbCr(CMY2RGB(*this));
  if (type=="TSL") return RGB2YCbCr(TSL2RGB(*this));
  if (type=="CIExy") return RGB2YCbCr(CIExy2RGB(*this));
  throw string(" GUColor::toYCbCr incorrect type of color");
}


////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toYUV() throw(string)
{
  if (type=="RGB") return RGB2YUV(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toYUV color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2YUV(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2YUV(YCbCr2RGB(*this));
  if (type=="YUV") return *this;
  if (type=="YIQ") return RGB2YUV(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2YUV(XYZD652RGB(*this));
  if (type=="Lab") return RGB2YUV(Lab2RGB(*this));
  if (type=="Luv") return RGB2YUV(Luv2RGB(*this));
  if (type=="CMY") return RGB2YUV(CMY2RGB(*this));
  if (type=="TSL") return RGB2YUV(TSL2RGB(*this));
  if (type=="CIExy") return RGB2YUV(CIExy2RGB(*this));
  throw string(" GUColor::toYUV incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toYIQ()  throw(string)
{
  if (type=="RGB") return RGB2YIQ(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toYIQ color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2YIQ(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2YIQ(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2YIQ(YUV2RGB(*this));
  if (type=="YIQ") return *this;
  if (type=="XYZD65") return RGB2YIQ(XYZD652RGB(*this));
  if (type=="Lab") return RGB2YIQ(Lab2RGB(*this));
  if (type=="Luv") return RGB2YIQ(Luv2RGB(*this));
  if (type=="CMY") return RGB2YIQ(CMY2RGB(*this));
  if (type=="TSL") return RGB2YIQ(TSL2RGB(*this));
  if (type=="CIExy") return RGB2YIQ(CIExy2RGB(*this));
  throw string(" GUColor::toYIQ incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toXYZD65()  throw(string)
{
  if (type=="RGB") return RGB2XYZD65(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toXYZD65 color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2XYZD65(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2XYZD65(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2XYZD65(YUV2RGB(*this));
  if (type=="YIQ") return RGB2XYZD65(YIQ2RGB(*this));
  if (type=="XYZD65") return *this;
  if (type=="Lab") return RGB2XYZD65(Lab2RGB(*this));
  if (type=="Luv") return RGB2XYZD65(Luv2RGB(*this));
  if (type=="CMY") return RGB2XYZD65(CMY2RGB(*this));
  if (type=="TSL") return RGB2XYZD65(TSL2RGB(*this));
  if (type=="CIExy") return RGB2XYZD65(CIExy2RGB(*this));
  throw string(" GUColor::toXYZD65 incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toLab()  throw(string)
{
  if (type=="RGB") return RGB2Lab(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toLab color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2Lab(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2Lab(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2Lab(YUV2RGB(*this));
  if (type=="YIQ") return RGB2Lab(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2Lab(XYZD652RGB(*this));
  if (type=="Lab") return *this;
  if (type=="Luv") return RGB2Lab(Luv2RGB(*this));
  if (type=="CMY") return RGB2Lab(CMY2RGB(*this));
  if (type=="TSL") return RGB2Lab(TSL2RGB(*this));
  if (type=="CIExy") return RGB2Lab(CIExy2RGB(*this));
  throw string(" GUColor::toLab incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toLuv()  throw(string)
{
  if (type=="RGB") return RGB2Luv(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toLuv color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2Luv(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2Luv(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2Luv(YUV2RGB(*this));
  if (type=="YIQ") return RGB2Luv(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2Luv(XYZD652RGB(*this));
  if (type=="Lab") return RGB2Luv(Lab2RGB(*this));
  if (type=="Luv") return *this;
  if (type=="CMY") return RGB2Luv(CMY2RGB(*this));
  if (type=="TSL") return RGB2Luv(TSL2RGB(*this));
  if (type=="CIExy") return RGB2Luv(CIExy2RGB(*this));
  throw string(" GUColor::toLuv incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::toCMY()  throw(string)
{
  if (type=="RGB") return RGB2CMY(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toCMY color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2CMY(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2CMY(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2CMY(YUV2RGB(*this));
  if (type=="YIQ") return RGB2CMY(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2CMY(XYZD652RGB(*this));
  if (type=="Lab") return RGB2CMY(Lab2RGB(*this));
  if (type=="Luv") return RGB2CMY(Luv2RGB(*this));
  if (type=="CMY") return *this;
  if (type=="TSL") return RGB2CMY(TSL2RGB(*this));
  if (type=="CIExy") return RGB2CMY(CIExy2RGB(*this));
  throw string(" GUColor::toCMY incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor  GUColor::toTSL()throw(string)
{
  if (type=="RGB") return RGB2TSL(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toCMY color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2TSL(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2TSL(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2TSL(YUV2RGB(*this));
  if (type=="YIQ") return RGB2TSL(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2TSL(XYZD652RGB(*this));
  if (type=="Lab") return RGB2TSL(Lab2RGB(*this));
  if (type=="Luv") return RGB2TSL(Luv2RGB(*this));
  if (type=="CMY") return RGB2TSL(CMY2RGB(*this));
  if (type=="TSL") return *this;
  if (type=="CIExy") return RGB2TSL(CIExy2RGB(*this));
  throw string(" GUColor::toTSL incorrect type of color");
}


////////////////////////////////////
//
//
////////////////////////////////////
GUColor  GUColor::toCIExy()throw(string)
{
  if (type=="RGB") return RGB2CIExy(*this);
  if (type=="NormalizedRGB")
      throw string(" GUColor::toCMY color is of type   NormalizedRGB so can not be translated.");
  if (type=="HSV") return RGB2CIExy(HSV2RGB(*this));
  if (type=="YCbCr")  return RGB2CIExy(YCbCr2RGB(*this));
  if (type=="YUV") return RGB2CIExy(YUV2RGB(*this));
  if (type=="YIQ") return RGB2CIExy(YIQ2RGB(*this));
  if (type=="XYZD65") return RGB2CIExy(XYZD652RGB(*this));
  if (type=="Lab") return RGB2CIExy(Lab2RGB(*this));
  if (type=="Luv") return RGB2CIExy(Luv2RGB(*this));
  if (type=="CMY") return RGB2CIExy(CMY2RGB(*this));
  if (type=="TSL") return RGB2CIExy(TSL2RGB(*this));
  if (type=="CIExy") return *this;
  throw string(" GUColor::toTSL incorrect type of color");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2HSV(GUColor c)throw (string)
{
 if (c.type!="RGB") throw string("GUColor::RGB2HSV c is not RGB");
 double R=c[0];
 double G=c[1];
 double B=c[2];
 double minVal=min(min(R, G), B);
 double V=max(max(R, G), B);
 double S,H;
 double Delta=V-minVal;
  // Calculate saturation: saturation is 0 if r, g and b are all 0
  if (V==0.0)   S=0.0;
  else  S=Delta/V;

  if (S==0.0)
    H=0.0; //   -- Achromatic: When s = 0, h is undefined but who cares
  else{      // -- Chromatic
    if (R==V) // -- between yellow and magenta [degrees]
      H=(60.0*(G-B))/Delta;
    else{
      if (G==V) // -- between cyan and yellow
        H=120.0+(60.0*(B-R))/Delta;
      else //-- between magenta and cyan
        H=240.0+(60.0)*(R-G)/Delta;
   }
  }

  if (H<0.0)  H=H+360.0 ;

  return GUColor((H*M_PI)/180., S, V/255.0,"HSV");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::HSV2RGB(GUColor c) throw (string)
{
 if (c.type!="HSV") throw string("GUColor::HSV2RGB c is not HSV");
 double H=(c[0]*180.)/M_PI;
 double S=c[1];
 double V=c[2];
 double R,G,B,hTemp,p,q,t,f;
 int i;

 if (S==0.0){ // color is on black-and-white center line
    R=V;//           -- achromatic: shades of gray
    G=V;//           -- supposedly invalid for h=0 but who cares
    B=V;
 }
 else{ //-- chromatic color
    if (H==360.0)// then  -- 360 degrees same as 0 degrees
      hTemp=0.0;
    else
      hTemp=H;


    hTemp=hTemp/60.0;//   -- h is now in [0,6)
    i=(int(hTemp));//  -- largest integer <= h
    f=hTemp-double(i);       //   -- fractional part of h

    p=V*(1.0-S);
    q=V*(1.0-(S*f));
    t=V*(1.0-(S*(1.0-f)));

    switch(i){
      case 0:
        R = V;
        G = t;
        B = p;
      break;
      case 1:
        R = q;
        G = V;
        B = p;
      break;
      case 2:
        R = p;
        G = V;
        B = t;
      break;
      case 3:
        R = p;
        G = q;
        B = V;
      break;
      case 4:
        R = t;
        G = p;
        B = V;
      break;
      case 5:
        R = V;
        G = p;
        B = q;
      break;
      };
  }
  return GUColor(R*255, G*255, B*255,"RGB");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2YCbCr(GUColor c)  throw (string)
{
if (c.type!="RGB") throw string("GUColor::RGB2YCbCr c is not RGB");
double Y = 0.29900*c[0] + 0.58700*c[1] + 0.11400*c[2];
double Cb = -0.16874*c[0] - 0.33126*c[1] + 0.50000*c[2];
double Cr = 0.50000*c[0]-0.41869*c[1] - 0.08131*c[2];
return GUColor(Y,Cb,Cr,"YCbCr");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::YCbCr2RGB(GUColor c)  throw (string)
{
if (c.type!="YCbCr") throw string("GUColor::YCbCr2RGB c is not YCbCr");
double R = c[0] + 1.40200*c[2];
double G = c[0] - 0.34414*c[1] - 0.71414*c[2];
double B = c[0] + 1.77200*c[1];
return GUColor(R,G,B,"RGB");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2NormalizedRGB(GUColor c) throw (string)
{
 if (c.type!="RGB") throw string("GUColor::RGB2NormalizedRGB c is not RGB");
 if (c[0]==c[1]==c[2]==0) return GUColor(0,0,0,"NormalizedRGB");
 double c0,c1,c2;
 c0=c[0]/(c[0]+c[1]+c[2]);
 c1=c[1]/(c[0]+c[1]+c[2]);
 c2=1-(c0+c1);
 return  GUColor(c0,c1,c2,"NormalizedRGB");
}
////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2YUV(GUColor c)throw (string)
{
 if (c.type!="RGB") throw string("GUColor::RGB2YUV c is not RGB");
 double Y = c[0] *  0.299 + c[1] *  0.587 + c[2] *  0.114;
 double U = c[0] * -0.169 + c[1] * -0.332 + c[2] *  0.500 + 128.;
 double V = c[0] *  0.500 + c[1] * -0.419 + c[2] * -0.0813 + 128.;
 return GUColor(Y,U,V,"YUV");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::YUV2RGB(GUColor c)throw (string)
{
 if (c.type!="YUV") throw string("GUColor::YUV2RGB c is not YUV");
 double  R = 1.0004*c[0] + -0.0002*c[1]+ (1.4017 * (c[2] - 128.0));
 double  G = 0.9994*c[0]-(0.3440* (c[1] - 128.0)) - (0.71139 * (c[2] - 128.0));
 double  B = 1.0018*c[0] + (1.7716 * (c[1] - 128.0))-0.0003*c[2];
 return GUColor(R,G,B,"RGB");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2YIQ(GUColor c)throw (string)
{
 if (c.type!="RGB") throw string("GUColor::RGB2YIQ c is not RGB");
 double  Y = 0.299*c[0] + 0.587*c[1] + 0.114*c[2];
 double  I = 0.596*c[0] - 0.274*c[1] - 0.322*c[2];
 double  Q = 0.211*c[0] - 0.523*c[1] + 0.312*c[2];
 return GUColor(Y,I,Q,"YIQ");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::YIQ2RGB(GUColor c)throw (string)
{
 if (c.type!="YIQ") throw string("GUColor::YIQ2RGB c is not YIQ");
 double  R = c[0] + 0.9562*c[1] + 0.6214*c[2];
 double  G = c[0] - 0.2727*c[1] - 0.6468*c[2];
 double  B = c[0] - 1.1037*c[1] + 1.7006*c[2];
  return GUColor(R,G,B,"RGB");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2XYZD65(GUColor c)throw (string)
{
 if (c.type!="RGB") throw string("GUColor::RGB2XYZD65 c is not RGB");
  double var_R = ( c[0] / 255. );        //R = From 0 to 255
  double var_G = ( c[1] / 255. );        //G = From 0 to 255
  double var_B = ( c[2] / 255. );        //B = From 0 to 255
  if ( var_R > 0.04045 ) var_R = pow(( ( var_R + 0.055 ) / 1.055 ),2.4);
  else                   var_R = var_R / 12.92;
  if ( var_G > 0.04045 ) var_G = pow(( ( var_G + 0.055 ) / 1.055 ),2.4);
  else                   var_G = var_G / 12.92;
  if ( var_B > 0.04045 ) var_B = pow(( ( var_B + 0.055 ) / 1.055 ),2.4);
  else                   var_B = var_B / 12.92;
  var_R = var_R * 100;
  var_G = var_G * 100;
  var_B = var_B * 100;

  //Observer. = 2°, Illuminant = D65
  double X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
  double Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
  double Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;
  return GUColor(X,Y,Z,"XYZD65");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::XYZD652RGB(GUColor c)throw (string)
{
 if (c.type!="XYZD65") throw string("GUColor::XYZD652RGB c is not XYZD65");
// double ref_X =  95.047;        //Observer = 2°, Illuminant = D65
// double ref_Y = 100.000;
// double ref_Z = 108.883;
 double var_X = c[0] / 100;        //X = From 0 to ref_X
 double var_Y = c[1] / 100;        //Y = From 0 to ref_Y
 double var_Z = c[2] / 100;        //Z = From 0 to ref_Y
 double var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
 double var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
 double var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;
 if ( var_R > 0.0031308 ) var_R = 1.055 * ( pow(var_R , ( 1. / 2.4 )) ) - 0.055;
 else                     var_R = 12.92 * var_R;
 if ( var_G > 0.0031308 ) var_G = 1.055 * ( pow(var_G , ( 1. / 2.4 )) ) - 0.055;
 else                     var_G = 12.92 * var_G;
 if ( var_B > 0.0031308 ) var_B = 1.055 * ( pow(var_B , ( 1. / 2.4 )) ) - 0.055;
 else                     var_B = 12.92 * var_B;
 double R = var_R * 255;
 double G = var_G * 255;
 double B = var_B * 255;
 return GUColor(R,G,B,"RGB");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2Lab(GUColor c)throw (string)
{
  if (c.type!="RGB") throw string("GUColor::RGB2Lab c is not RGB");
  GUColor XYZD65=c.toXYZD65();

  double var_X = XYZD65[0] /  95.047;          //Observer = 2°, Illuminant = D65
  double var_Y = XYZD65[1] / 100.000;
  double var_Z = XYZD65[2] / 108.883;

  if ( var_X > 0.008856 ) var_X = pow(var_X , ( 1./3. ));
  else                    var_X = ( 7.787 * var_X ) + ( 16 / 116 );
  if ( var_Y > 0.008856 ) var_Y = pow(var_Y , ( 1./3. ));
  else                    var_Y = ( 7.787 * var_Y ) + ( 16 / 116 );
  if ( var_Z > 0.008856 ) var_Z = pow(var_Z , ( 1./3. ));
  else                    var_Z = ( 7.787 * var_Z ) + ( 16 / 116 );

  double L = ( 116 * var_Y ) - 16;
  double a = 500 * ( var_X - var_Y );
  double b = 200 * ( var_Y - var_Z );
  return GUColor(L,a,b,"Lab");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::Lab2RGB(GUColor c)throw (string)
{
  if (c.type!="Lab") throw string("GUColor::Lab2RGB c is not Lab");
  //pasamos a XYZ D65
  double var_Y = ( c[0] + 16 ) / 116;
  double var_X = c[1] / 500 + var_Y;
  double var_Z = var_Y - c[2] / 200;
  if ( pow( var_Y,3.) > 0.008856 ) var_Y = pow(var_Y,3);
  else                      var_Y = ( var_Y - 16 / 116 ) / 7.787;
  if ( pow(var_X,3.) > 0.008856 ) var_X = pow(var_X,3);
  else                      var_X = ( var_X - 16 / 116 ) / 7.787;
  if ( pow(var_Z,3.) > 0.008856 ) var_Z = pow(var_Z,3);
  else                      var_Z = ( var_Z - 16 / 116 ) / 7.787;

  double X = 95.047 * var_X ;    //ref_X =  95.047  Observer= 2°, Illuminant= D65
  double Y = 100.000 * var_Y ;    //ref_Y = 100.000
  double Z = 108.883 * var_Z;     //ref_Z = 108.883
  //pasamos a RGB
  GUColor XYZD65(X,Y,Z,"XYZD65");
  return  XYZD65.toRGB();
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2Luv(GUColor c)throw (string)
{
  if (c.type!="RGB") throw string("GUColor::RGB2Luv c is not RGB");
  
  if (c[0]==c[1]==c[2]==0) return GUColor(0,0,0,"Luv");
  
  GUColor XYZD65=c.toXYZD65();
  
  
  double var_U = ( 4. * XYZD65[0] ) / ( XYZD65[0] + ( 15. * XYZD65[1] ) + ( 3. * XYZD65[2] ) );
  double var_V = ( 9. * XYZD65[1] ) / ( XYZD65[0] + ( 15. * XYZD65[1] ) + ( 3. * XYZD65[2] ) );
  double var_Y = XYZD65[1] / 100.;
  if ( var_Y > 0.008856 ) var_Y = pow(var_Y , ( 1./3. ));
  else                    var_Y = ( 7.787 * var_Y ) + ( 16. / 116. );
  //Observer= 2°, Illuminant= D65
  double ref_X =  95.047;
  double ref_Y = 100.000;
  double ref_Z = 108.883;
  double ref_U = ( 4 * ref_X ) / ( ref_X + ( 15 * ref_Y ) + ( 3 * ref_Z ) );
  double ref_V = ( 9 * ref_Y ) / ( ref_X + ( 15 * ref_Y ) + ( 3 * ref_Z ) );

  double L = ( 116 * var_Y ) - 16;
  double u = 13 * L * ( var_U - ref_U );
  double v = 13 * L * ( var_V - ref_V );

  return GUColor(L,u,v,"Luv");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::Luv2RGB(GUColor c)throw (string)
{
  if (c.type!="Luv") throw string("GUColor::Luv2RGB c is not Luv");

  double var_Y = ( c[0] + 16 ) / 116;
  if ( pow(var_Y,3.) > 0.008856 ) var_Y = pow(var_Y,3);
  else                      var_Y = ( var_Y - 16 / 116 ) / 7.787;
     //Observer= 2°, Illuminant= D65
  double ref_X =  95.047;
  double ref_Y = 100.000;
  double ref_Z = 108.883;
  double ref_U = ( 4 * ref_X ) / ( ref_X + ( 15 * ref_Y ) + ( 3 * ref_Z ) );
  double ref_V = ( 9 * ref_Y ) / ( ref_X + ( 15 * ref_Y ) + ( 3 * ref_Z ) );

  double var_U = c[1] / ( 13 * c[0] ) + ref_U;
  double var_V = c[2] / ( 13 * c[0] ) + ref_V;
  double Y = var_Y * 100;
  double X =  - ( 9 * Y * var_U ) / ( ( var_U - 4 ) * var_V  - var_U * var_V );
  double Z = ( 9 * Y - ( 15 * var_V * Y ) - ( var_V * X ) ) / ( 3 * var_V );
  GUColor XYZD65(X,Y,Z,"XYZD65");
  return XYZD65.toRGB();
}
////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2CMY(GUColor c)throw (string)
{
  if (c.type!="RGB") throw string("GUColor::RGB2CMY c is not RGB");
 //RGB values = From 0 to 255

  double C = 1. - ( c[0] / 255. );
  double M = 1. - ( c[1] / 255. );
  double Y = 1. - ( c[2] / 255. );
  return GUColor(C,M,Y,"CMY");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::CMY2RGB(GUColor c)throw (string)
{
  if (c.type!="CMY") throw string("GUColor::CMY2RGB c is not CYM");
 //RGB values = From 0 to 255

//CMY values = From 0 to 1

  double R = ( 1. - c[0] ) * 255.;
  double G = ( 1. - c[1] ) * 255.;
  double B = ( 1. - c[2] ) * 255.;
  return GUColor(R,G,B,"RGB");
}

////////////////////////////////////
//
//
////////////////////////////////////
bool GUColor::isValidType(string Type)
{
 if (Type!="RGB" && Type!="HSV"&& Type!="YCbCr" && Type!="NormalizedRGB" && Type!="YUV"
  && Type!="YIQ" && Type!="XYZD65" && Type!="Lab" && Type!="Luv" && Type!="CMY" && Type!="TSL" && Type!="CIExy")
   return false;
 return true;
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2TSL(GUColor c)throw (string)
{
	if (c.type!="RGB") throw string("GUColor::RGB2TSL c is not RGB");
	double r=c[0]/(c[0]+c[1]+c[2]);
	double g=c[1]/(c[0]+c[1]+c[2]);
	double rprima=r-(1./3.);
	double gprima=g-(1./3.);
	double S=std::sqrt( double(9./5.)*(rprima*rprima+gprima*gprima));
	double T=0;
	if (gprima>0) T=(atan(rprima/gprima) / (2.*M_PI)) + 1./4.;
	if (gprima<0) T=(atan(rprima/gprima) / (2.*M_PI)) + 3./4.;
	if (gprima==0) T=0;
	double L=0.299*c[0]+0.587*c[1]+0.114*c[2];
	return GUColor(T,S,L,"TSL");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::TSL2RGB(GUColor c)throw (string)
{
	if (c.type!="TSL") throw string("GUColor::TSL2RGB c is not TSL");
	throw string("TSL2RGB convert from TSL to RGB is not implemented");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::RGB2CIExy(GUColor c)throw (string)
{
	if (c.type!="RGB") throw string("GUColor::RGB2TSL c is not RGB");
	GUColor XYZ=c.toXYZD65();
	double x=XYZ[0]/(XYZ[0]+XYZ[1]+XYZ[2]);
	double y=XYZ[1]/(XYZ[0]+XYZ[1]+XYZ[2]);
	double z=XYZ[2]/(XYZ[0]+XYZ[1]+XYZ[2]);
	return GUColor(x,y,z,"CIExy");
}

////////////////////////////////////
//
//
////////////////////////////////////
GUColor GUColor::CIExy2RGB(GUColor c)throw (string)
{
	if (c.type!="CIExy") throw string("GUColor::CIExy2RGB c is not TSL");
	throw string("CIExy2RGB convert from CIExy to RGB is not implemented");
}

////////////////////////////////////
//
//
////////////////////////////////////
double GUColor::toInt(double f)
{
 double rest= f - double(int(f));
 if (rest>0.5) return double(int(f)+1);
 else return double(int(f));
}

GUColor GUColor::to(string color) throw(string)
{
	if ( color=="RGB") return toRGB();
	if ( color=="NormalizedRGB") return toNormalizedRGB();
	if ( color=="HSV") return toHSV();
	if ( color=="YCbCr") return toYCbCr();
	if ( color=="YUV") return toYUV();
	if ( color=="YIQ") return toYIQ();
	if ( color=="XYZD65") return toXYZD65();
	if ( color=="Lab") return toLab();
	if ( color=="Luv") return toLuv();
	if ( color=="CMY") return toCMY();
	if ( color=="TSL") return toTSL();
	if ( color=="CIExy") return toCIExy();
	
	throw string(" GUColor::to, unrecignized space color "+color);
}
