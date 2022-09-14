# Repository description
Repository contains 237 tests and 11 notebooks showing peculiarities of popular 
among data scientists tools. Most interesting parts are:
1. Python language itself. Topics included:
   1. Parallelism: threads, processes, coroutines
   2. Decorators
   3. `match` statement
   4. Runtime check-able type hints
   5. Import of distributed namespace package
   6. unittests: custom test case, test skipping, custom type comparator
2. Tensorflow:
   1. Auto-differentiation
   2. Saving modules
   3. Operations with Tensor/Variable
   4. Keras:
      1. Saving keras model at different stages of it's lifecycle
      2. Masking and padding
      3. Data preprocessing layers
      4. Implementation of custom `Dense` layer and learning model using it
      5. Custom: callback, fit, loss, metric
      6. Usage of `Dataset` object
      7. Usage of `layer.trainable` param
3. PyTorch:
   1. Implementation of custom `Dense` layer and learning model using it
   2. Usage of `DataLoader` and custom `Dataset`
   3. Saving PyTorch model
4. OpenCV:
   1. Basics: image read modes, rotation, translation
   2. Blobs detection
   3. Thresholding
   4. Canny edge detection
   5. Contour detection
   6. Image blurring
   7. Template matching
   8. Playing video from the camera
   9. Mouse events handling
5. Pandas:
   1. Series: 
      1. Labels alignment
      2. Rename
      3. `dt`, `str`, `sparse` accessors
      4. Nullable `boolean` data type
   2. DataFrame:
      1. Creation peculiarities
      2. Columns assignment
      3. Series broadcasting
      4. Combining
      5. Aggregation
      6. Transformation
      7. Interaction with scipy sparce matrices
6. Numpy
   1. ND-arrays features: masking, new axis creation by `None`, broadcasting, 
boolean/integer indexing, matrix product
   2. `__array_interface__` interaction
   3. Structured data types and record arrays
   4. Custom array containers and handling `numpy` functions operation on it
   5. Subclassing ND-arrays
   6. Miscellaneous: `np.nan` and `np.inf` behaviour, floating point errors behaviour
7. Matplotlib
   1. Simple chart example with setting some properties
   2. Interactive chart using native jupyter GUI for matplotlib
8. Scipy
   1. Filtering functions: correlate1d, convolve1d
   2. Image labeling, object detection, object size measurement

# Environment
Create environment:

```conda env create --prefix ./cenv --file ./environment.yml```

Activate environment:

```conda activate ./cenv```

# Run configurations
Repository contains 4 run/debug configuration for PyCharm/DataSpell ides stored in
./idea/runConfigurations:
1. test_main - unittest run configuration to run all tests except for tensorflow tests 
(time < 3s)
2. tf_test_main - unittest run configuration to run only tensorflow tests (time ~ 15s)
3. test_main_manual - python run configuration to run all tests except for tensorflow tests 
(time < 3s)
4. tf_test_main_manual - python run configuration to run only tensorflow tests (time ~ 15s)

We need run/debug configurations 3-4 because sometimes DataSpell refuses to run 
unittest run/debug configurations

# PyLint
PyLint is configured for this repository