# Hough line
Find lines in the image using Hough transform.

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
cp -r ../../../computer_vision_basics_data/hough_line/build/* .
```

Run hough line detection:
```
./hough_line <image_file>
```
