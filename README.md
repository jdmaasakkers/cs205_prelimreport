# Parallelized Image Recognition in Spark + MPI
## CS205 - Interim progress report

Image recognition is a hot topic in machine learning communities. Our project is to apply statistical learning theory in the Spark framework to classify images. We further aim to optimize the training of our learning algorithm by implementing MPI parallelization. We will explore both model and data parallelization as many examples of labeled images are critical to training an accurate model. We will start with a common dataset of handwritten digits (MNIST), and will set up a framework where custom images can be imported.

### Project Goals
- Implement both serial, OpenMPI+MPI, and Spark regularized statistical learning algorithms. This will initially take the form of a linear classifier (e.g. 1-layer neural network) with L-2 loss.
- Analyze and benchmark data vs. model parallelization approaches. Weigh the benefits of computation cost vs. testing accuracy.
- Build a framework where custom images can be imported.

### Serial implementation

The code used for this is included in [Code_Serial.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Serial.py).
![Serial-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Sizes_Serial.png)


### OpenMP parallalization
OpenMP parallalization is implented using Cython. The inner loops of the learning algorithms are parallalized using nogil pranges that  use static scheduling. By just parallalizing the inner loops using OpenMP, the outer loops can later be parallalized using MPI. The code is compiled using GCC (5.2.0) and benchmarked for different numbers of cores. We find that using Cython with just one thread (so no parallalization) already gives a large speedup compared to the regular serial code. For all problem sizes, we find speedups around 66 compared to the serial code. This speedup is due to ability of Cython to run in c. The figure below shows runtimes for different problem sizes (number of samples) and different cores.

![OpenMP-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Runtime_OpenMP.png)

The Figure shows lower runtimes for an increasing number of cores. The Figure below shows the associaed speedup and scaled speedup using $n = 1000$ as the base case. 
The used Python script is [Code_OpenMP.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_OpenMP.py) together with Cython module [train_ml_prange.pyx](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/train_ml_prange.pyx).

![OpenMP-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_OpenMP.png)

### OpenMP + MPI parallalization

### Spark parallalization

The Spark version of the code is [Code_Spark.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Spark.py).
![Spark-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_Spark.png)

Python code used for all the plots is included in the code directory. 

### Future work
We will continue work along three different avenues:
- Optimize the hybrid parallization using OpenMP + MPI and do a more rigurous benchmark including running on 8 compute nodes on Odyssey.
- Optimize the Spark parallalization for AWS, implement GPU acceleration, and benchmark for different setups. We will also add visuals illustrating the workflow. 
- Build a framework where custom images can be imported into the learning algorithm. 
