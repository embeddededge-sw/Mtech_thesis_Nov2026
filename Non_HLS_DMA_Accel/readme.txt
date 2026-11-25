

Command to run the test: -
=========================================

Usage: - python3 cnn_benchmark.py <path of bmp/ folder having test images and label.txt> <no of images to test>

Command: - 
sudo -i
source /etc/profile.d/pynq_venv.sh
cd /home/ubuntu/nonhls_purertl_dma_accel/
python3 cnn_benchmark.py bmp/ 5000
exit

Typical O/p - 
..................
  4989  train_12.bmp                     3          3  OK
  4990  train_2.bmp                      4          4  OK
  4991  train_9.bmp                      4          4  OK
  4992  train_0.bmp                      5          5  OK
  4993  train_11.bmp                     5          5  OK
  4994  train_13.bmp                     6          6  OK
  4995  train_18.bmp                     6          6  OK
  4996  train_15.bmp                     7          7  OK
  4997  train_17.bmp                     8          8  OK
  4998  train_4.bmp                      9          9  OK
  4999  train_19.bmp                     9          9  OK

================================================================
Total inferences : 5000
Correct          : 5000
Accuracy         : 100.00 %
Elapsed time     : 1074833137 ns
Elapsed time     : 1.074833137 s
================================================================


Note: -
The folder - "nonhls_purertl_dma_accel/" must contain below files for it to work with python script.
cnn_benchmark.py, cnn_mnist.bit, cnn_mnist.hwh and bmp/ folder. The bmp/ folder contains all images of digits and label.txt.
