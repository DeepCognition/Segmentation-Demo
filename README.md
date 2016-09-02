# Setting up Segmentaton Demo on web
    
Step by Step procedure to set up Image segmentation demo on your website. We used pre-trained vgg16 model based on Caffe to segment the images. The base code and trained weights are taken from [here](https://bitbucket.org/aquariusjay/deeplab-public-ver2)
In this explanation we assume familiarity with deeplab version of Caffe. If you need more information about the same, please go through DEEPLAB_README.md
Below are the steps to be followed to run the segmentation task.
    
### Download and Install Deeplab version of Caffe

1) Clone the base code from the deeplab version-2 repository.
2) create a Makefile.config similar to Makefile.config.example.
3) Make changes to Makefile and Makefile.config files to successfully build the source code. (Refer to diff between commits "deeplab caffe ver2 no change" and "modification to successfully build" for the details. Note that we used CPU_ONLY build.)

### Download the pre-trained weights and model files

We used the pre-trained model and corresponding prototxt files for deeplab_v2 vgg16 model from [here](http://liangchiehchen.com/projects/DeepLabv2_vgg.html)
The download contains a zip file containing many files. We use train_iter_20000.caffemodel for pre-trained weights and test.prototxt as the network. (We are not uploading the weights file due to size limitations of github)

Make changes to test.prototxt to replace the variables with constants. Below are the few mandatory changes:
${NET_ID} --> Any name you would want to use for the network
${DATA_ROOT} --> Valid path which is later used by the network to search for input images.
${EXP} --> voc12 (This should not be changed, network will not work if we use any other name)
${NUM_LABELS} --> 21
Add Softmax layer after generating the 21 channel output from convolutional layer to convert the values to probabilities. [0-1)

Check the diff between commits "added vgg model files from deeplab website" and "modified network to run segmentation test" for full range of changes to make.

### Generating Segmentation binary

The source code has few examples for classification which will be compiled while building caffe and their binaries can be used to classify images. Similar operation can be done for Segmentation.
We have written a simple cpp file for this purpose. The logic for this file is as follows:

1) It waits indefinitely looking at a particular directory for input images.
2) As soon as it sees a file in the image directory, it will check if it's a valid file and if yes it will write the filename to list.txt (list.txt is used by our prototxt file to look for names of the input images to process)
3) On writing valid image name to list.txt file, the network is started and runs for one forward propogation throug all the layers. Take the output of the network and perform few postprocessing steps to generate segmented output image.

Refer to examples/segmentation/segmentation.cpp for more detailed understanding. Build the Caffe again to include newly added file.

### Actual Segmentation Task

Before we proceed to segment the image we need to create a text file containing labels for the segments.(Refer to models/segmentation/labels.txt). Once this is done we can segment any jpeg image as below:

1) Run the command in the shell: "sudo .build_release/examples/segmentation/segmentation.bin models/segmentation/test.prototxt models/segmentation/train_iter_20000.caffemodel labels.txt segment"
2) copy the image to be segmented in "/home/ubuntu/segment/" directory
3) Wait for segmentation to complete and observe the output generated at "/tmp/segment/"

#### This can be easily integrated to the web UI to get an image from the user, post the image to backend and return the segmented image as output. Please check live demo on our [website](www.deepcognition.ai).

### Credits

We would like to extend our sincere thanks to [liang-Chieh Chen](liangchiehchen.com) for making the deeplab_v2 and pretrained models publicly available to use. 