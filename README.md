Source Code for Machine Perception Course Project at ETH (263-3710-00L)

# Dynamic Gesture Recognition Project Skeleton Code
Visit [here](https://ait.ethz.ch/teaching/courses/2018-SS-Machine-Perception/) for more information about the Machine Perception course.

All questions should first be directed to our course Piazza before being sent to my [e-mail address](mailto:eaksan@inf.ethz.ch).

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

1. Install dependencies by running (with `sudo` appended if necessary)
```
python3 setup.py install
```
2. Download training, validation and test datasets from kaggle project page.
3. Update data and output paths in `config.py`.
4. Train the model provided with source code by running 
```
python3 training.py
```
4. When your model has completed training, it will perform a full evaluation on the test set. This output can be found in the folder `runs/<experiment_name>/` as `submission_<experiment_name>.csv`.

Submit this `csv` file to our page on [Kaggle](https://www.kaggle.com/c/mp18-dynamic-gesture-recognition/submissions).

## Remarks
You can use external libraries. Make sure that you add them into `setup.py` before submitting your code.

