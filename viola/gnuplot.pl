
perl ./driveGnuPlotStreams.pl 8 3 \
50 50 50 \
0 1 1000 5000 0 1000 \
1500x340+0+0 1500x3400+0+0 1500x340+0+0 \
'total' 'color' 'shape' 'torso_color' 'z' 'found' 'Z' 'ZE' \
0 0 0 0 0 0 1 2

./particle_filter_main 200 60 60 | \
perl ./driveGnuPlotStreams.pl 'STREAMP' 8 3 \
50 50 50 \
0 2 1000 5000 0 1000 \
1500x340+0+0 1500x340+0+0 1500x340+0+0 \
'total' 'color h' 'shape h' 'color t' 'z h' 'found' 'Z' 'ZE' \
0 0 0 0 0 0 1 2

./particle_filter_main 200 60 60 | \
perl ./driveGnuPlotStreams.pl 'STREAM' 8 3 \
50 50 50 \
0 2 1000 5000 0 1000 \
1500x340+0+0 1500x340+0+0 1500x340+0+0 \
'total' 'color h' 'shape h' 'color t' 'z h' 'found' 'Z' 'ZE' \
0 0 0 0 0 0 1 2
