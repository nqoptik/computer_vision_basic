# Image segmentation by using graphcuts algorithm and watershed algorithm
Using graphcuts algorithm and wateshed algorithm to segment images.

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
cp -r ../../../computer_vision_anlab_data/image_segmentation/build/* .
```

Run graphcuts segmentation:
```
./graphcuts_segmentation
```

Run watershed segmentation:
```
./watershed_segmentation
```
