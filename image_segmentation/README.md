# Image segmentation by using grabcut algorithm and watershed algorithm
Using grabcut algorithm and wateshed algorithm to segment images.

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
cp -r ../../../computer_vision_basics_data/image_segmentation/build/* .
```

Run grabcut segmentation:
```
./grabcut_segmentation
```

Run watershed segmentation:
```
./watershed_segmentation
```
