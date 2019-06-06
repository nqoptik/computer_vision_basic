# Detect sudoku board on images

## Build project
Build project with cmake:
```
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run project
Copy test data to build folder:
```
cp -r ../../../computer_vision_basics_data/sudoku_detection/build/* .
```

Run sudoku detection:
```
./sudoku_detection <image_file>
```
