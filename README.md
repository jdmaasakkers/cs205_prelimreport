# Parallelized Image Recognition in Spark + MPI
## CS205 - Interim progress report

Image recognition is a hot topic in machine learning communities. Our project is to apply statistical learning theory in the Spark framework to classify images. We further aim to optimize the training of our learning algorithm by implementing MPI parallelization. We will explore both model and data parallelization as many examples of labeled images are critical to training an accurate model. We will start with a common dataset of handwritten digits (MNIST), and will set up a framework where custom images can be imported.

### Project Goals
- Implement both serial, OpenMPI+MPI, and Spark regularized statistical learning algorithms. This will initially take the form of a linear classifier (e.g. 1-layer neural network) with L-2 loss.
- Analyze and benchmark data vs. model parallelization approaches. Weigh the benefits of computation cost vs. testing accuracy.
- Build a framework where custom images can be imported.

### Serial implementation
![Serial-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Sizes_Serial.png)


### OpenMP parallalization
![OpenMP-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Runtime_OpenMP.png)
![OpenMP-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_OpenMP.png)

### OpenMP + MPI parallalization

### Spark parallalization
![Spark-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_Spark.png)

### Future work
We will continue work along three different avenues:
- Optimize the hybrid parallization using OpenMP + MPI and do a more rigurous benchmark including running on 8 compute nodes on Odyssey.
- Optimize the Spark parallalization for AWS, implement GPU acceleration, and benchmark for different setups.
- Build a framework where custom images can be imported into the learning algorithm. 
