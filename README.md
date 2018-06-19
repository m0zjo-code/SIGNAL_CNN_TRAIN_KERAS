
# Signal Doctor - Neural Network Toolkit
This repo is an addition to the signaldoctor classification framework. Here scripts are present that allow the user to train CNNs.

The report is available here:
- https://goo.gl/PK8S3v

Data is provided in the NumPy archive format *.npz files, pre-formatted using the ``generate_training_data.py`` script in the signaldoctor framework. All of the scripts take the same inputs.

``` sh
python3 CNN_1D_Single.py --input /your/training/data.npz --prefix myamazingnetwork
```
The network prefix allows you to easily separate your output files if you have trained many similar networks.

``CNN_1D_Single.py`` - CNN for 1D features as specified in the report
``CNN_2D_Single.py`` - CNN for 2D features as specified in the report
``CNN_2D_Xfer.py`` - Transfer learning CNN for 2D features as specified in the report
