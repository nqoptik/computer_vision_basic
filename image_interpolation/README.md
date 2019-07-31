# Image interpolation
Image interpolation algorithms for image resizing and rotating.

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
cp -r ../../../computer_vision_basics_data/image_interpolation/build/* .
```

Run the image interpolation:
```
./image_interpolation <image_file>
```
