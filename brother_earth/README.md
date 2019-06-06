# Brother Earth project
A simple sense detector for Brother Earth project.

## Prerequisites
Install fftw3 library:
```
sudo apt-get install libfftw3-dev
```

## Build project
Build project with cmake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run project
Copy test data to build folder:
```
cp -r ../../../computer_vision_basics_data/brother_earth/build/* .
```

Run cross light detection:
```
./cross_light_detection
```

Run fast Hough circle detection:
```
./fast_hough_circle
```

Run orientation detection:
```
./orientation_fft
```
