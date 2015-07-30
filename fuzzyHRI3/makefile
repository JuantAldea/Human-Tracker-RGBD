LIBS=-lguia_addon -lgustereotools -lgustereo3 -lguopencv  -lpnmutils  `pkg-config --libs opencv`
CXXFLAGS= -g3 -I/usr/include -I.

all: main.cpp smileestimator.cpp imagecomposer.cpp
	g++ $(CXXFLAGS)  smileestimator.cpp imagecomposer.cpp main.cpp -o main $(LIBS) -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter

clean:
	rm *~ *.o main -f
