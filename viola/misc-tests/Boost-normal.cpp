
// normal_misc_examples.cpp

// Copyright Paul A. Bristow 2007, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of using normal distribution.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[normal_basic1
/*`
First we need some includes to access the normal distribution
(and some std output of course).
*/
#include <boost/math/distributions/normal.hpp> // for normal_distribution
#include <iostream>
using namespace std;

int main()
{
    boost::math::normal_distribution<double> s(0, 0.4); // (default mean = zero, and standard deviation = unity)

    float z = 0.1;
    //cout << "Area for z = " << z << " is " << cdf(s, z) << endl; // to get the area for z.
    //cout << "Area for z = " << -z << " is " << cdf(s, -z) << endl; // to get the area for z.

    cout << "Area for [+-z] = " << -z << " is " << 1 - (cdf(s, z) - cdf(s, -z)) << endl; // to get the area for z.
    cout << "Area for half = " << z << " is " << 1 - (2 * cdf(s, z) - 1)<< endl; // to get the area for z.
    return 0;
}
