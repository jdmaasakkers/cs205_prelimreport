# Parallelized Image Recognition in Spark + MPI
## CS205 - Interim progress report

Image recognition is a hot topic in machine learning communities. Our project is to apply statistical learning theory in the OpenMPI and Spark frameworks to classify images. We further aim to optimize the training of our learning algorithm by implementing MPI parallelization. We will explore both model and data parallelization as many examples of labeled images are critical to training an accurate model.

### Learning algorithm


The current implementation uses the MNIST database [(LeCun et al. 1998)](http://yann.lecun.com/exdb/mnist/) that consists handwritten digits (0-9). The database includes a training set of 60,000 and a test set of 10,000 each consisting of 28 by 28 pixels. Each pixel has a value between 0 and 255 (white to black). If time permits, we are planning to expand our model to import our own images (see 'Future Work'). 

### Project Goals
- Implement both serial, OpenMPI+MPI, and Spark regularized statistical learning algorithms. This will initially take the form of a linear classifier (e.g. 1-layer neural network) with L-2 loss.
- Analyze and benchmark data vs. model parallelization approaches. Weigh the benefits of computation cost vs. testing accuracy.
- Build a framework where custom images can be imported.

### Serial implementation
We first benchmark the serial implementation on Odyssey for different problem sizes. The code used for this is included in [Code_Serial.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Serial.py). As shown in the Figure below, we find that runtime scales linearly with the number of samples studied. 

![Serial-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Sizes_Serial.png)

### OpenMP parallalization
OpenMP parallalization is implented using Cython. The inner loops of the learning algorithms are parallalized using nogil pranges that  use static scheduling. By just parallalizing the inner loops using OpenMP, the outer loops can later be parallalized using MPI. The code is compiled using GCC (5.2.0) and benchmarked for different numbers of cores. We find that using Cython with just one thread (so no parallalization) already gives a large speedup compared to the regular serial code. For all problem sizes, we find speedups around 66 compared to the serial code. This speedup is due to ability of Cython to run in c. The figure below shows runtimes for different problem sizes (number of samples) and different cores.

![OpenMP-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Runtime_OpenMP.png)

The Figure shows lower runtimes for an increasing number of cores. The Figure below shows the associaed speedup and scaled speedup using *n = 1000* as the base case. We find that using both 5 and 10 cores leads to regular speedup above 1. However, using 10 cores is slower than using 5 cores due to additional overhead. However, when looking at the scaled speedup (increasing the problem size with the same ratio as increasing the number of processors), we do find that using 10 cores has the highest speedup. The brown line shows the efficiencies obtained with the different numbers of cores. Efficiencies are relatively large as we use a number of different pranges, each with their own overhead/allocation costs. As the problem size is relatively small, these overheads are relatively large. In addition to that, part of the module is not parallalized, leaving part of the code to be run in serial in all cases. The used Python script is [Code_OpenMP.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_OpenMP.py) together with Cython module [train_ml_prange.pyx](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/train_ml_prange.pyx).

![OpenMP-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_OpenMP.png)

### OpenMP + MPI parallalization
On top of the inner loop parallalization using OpenMP, we now implement MPI parallalization on the outer loop. 

### Spark parallalization
Spark allows a different method of parallalizing the learning algorithm. Using functional parallelism, Spark works using compositions of functions. We implement a Spark version of our code on the Amazone Web Services (AWS) EMR Spark cluster. We run our code with 1 master and 2 core nodes and validated that it gave the same results as the serial implementation. The resulting speedup compared to runnning the serial Odyssey code is shown in the figure below.

![Spark-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_Spark.png)

We find speedup larger than 1 for all problem sizes studied. More work will be done on applying Spark to additional lops of the algorithm and benchmarking it for varying hardware setups on AWS. The Spark version of the code is [Code_Spark.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Spark.py).
Python code used for all the plots is included in the code directory. 

### Future work
We will continue work along three different avenues:
- Optimize the hybrid parallization using OpenMP + MPI and do a more rigurous benchmark including running on 8 compute nodes on Odyssey.
- Optimize the Spark parallalization for AWS (adding Spark to additional loops in the program), implement GPU acceleration, and benchmark for different setups. We will also add visuals illustrating the workflow. 
- Build a framework where custom images can be imported into the learning algorithm. 
