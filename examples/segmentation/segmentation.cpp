#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/inotify.h>
#include <iostream>
#include <fstream>
#include <time.h>

#ifdef USE_OPENCV

#define EVENT_SIZE (sizeof(struct inotify_event))
#define BUF_LEN    (1024*(EVENT_SIZE + 16))

uint8_t color[21][3] = {{0, 0, 0},
			{128, 0, 0},
			{0, 128, 0},
			{128, 128, 0},
			{0, 0, 128},
			{128, 0, 128},
			{0, 128, 128},
			{128, 128, 128},
			{64, 0, 0},
			{192, 0, 0},
			{64, 128, 0},
			{192, 128, 0},
			{64, 0, 128},
			{192, 0, 128},
			{64, 128, 128},
			{192, 128, 128},
			{0, 64, 0},
			{128, 64, 0},
			{0, 192, 0},
			{128, 192, 0},
			{0, 64, 128}};
                                   
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class Segmenter {
 public:
  Segmenter(const string& model_file,
             const string& trained_file,
             const string& label_file);

  cv::Mat Segment();

 private:

  std::vector<cv::Mat> Predict();

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
};

Segmenter::Segmenter(const string& model_file,
                       const string& trained_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  //CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  input_geometry_ = cv::Size(513, 513);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

cv::Mat Segmenter::Segment() {
  std::vector<cv::Mat> output = Predict();

  int N = labels_.size();
  for(int i = 1; i < N; i++){
      cv::addWeighted(output[0], 1.0, output[i*3], 1.0, 0.0, output[0]);
      cv::addWeighted(output[1], 1.0, output[i*3+1], 1.0, 0.0, output[1]);
      cv::addWeighted(output[2], 1.0, output[i*3+2], 1.0, 0.0, output[2]); 
  }
  
  std::vector<cv::Mat> channels;
  channels.push_back(output[0]);
  channels.push_back(output[1]);
  channels.push_back(output[2]);

  cv::Mat out;
  cv::merge(channels, out);

  return out;
}

std::vector<cv::Mat> Segmenter::Predict(){
  std::vector<cv::Mat> segments;
  net_->ForwardPrefilled();
  Blob<float>* output_layer = net_->output_blobs()[0];
  float* data = output_layer->mutable_cpu_data();
  int num_channels = output_layer->channels();
  
  for(int i = 0; i < num_channels; i++){
    cv::Mat segment_(input_geometry_.height, input_geometry_.width, CV_32FC1, data);
    
    cv::Mat segment(input_geometry_.height, input_geometry_.width, CV_8UC1);
    cv::threshold(segment_, segment, 0.5, color[i][0], cv::THRESH_BINARY);
    segments.push_back(segment);
    
    cv::Mat segment1(input_geometry_.height, input_geometry_.width, CV_8UC1);
    cv::threshold(segment_, segment1, 0.5, color[i][1], cv::THRESH_BINARY);
    segments.push_back(segment1);
    
    cv::Mat segment2(input_geometry_.height, input_geometry_.width, CV_8UC1);
    cv::threshold(segment_, segment2, 0.5, color[i][2], cv::THRESH_BINARY);
    segments.push_back(segment2);
    
    data += input_geometry_.height * input_geometry_.width;
  }
  return segments;
}

int main(int argc, char** argv) {
  int length, i = 0;
  int fd;
  int wd[2];
  char buffer[BUF_LEN];

  fd = inotify_init();

  if ( fd < 0 ) {
      perror( "inotify_init" );
  }
  
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " labels.txt model" << std::endl;
    return 1;
  }
  
  string model = argv[4];
  string image_dir = string("/home/ubuntu/") + model;
  string out_dir = string("/tmp/") + model;

  wd[0] = inotify_add_watch(fd, image_dir.c_str(), IN_CLOSE_WRITE);

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string label_file   = argv[3];

  while(1){
	struct inotify_event *evt;
	length = read(fd, buffer, BUF_LEN);
	if(length < 0){
	  continue;
	}

	evt = (struct inotify_event*)&buffer[i];
	if(evt->len){
	  printf("evt->name = %s evt->mask = %x", evt->name, evt->mask);
	  if(evt->mask & IN_CLOSE_WRITE){
	    if(evt->mask & IN_ISDIR){
		printf("The directory %s was created\n", evt->name);
	    }else{
		time_t timer1;
		time_t timer2;

		time(&timer1);
		string file = evt->name;
		string infile = image_dir + "/" + file;
		string inlist = out_dir + "/list.txt";
		std::ofstream inp_list;
		inp_list.open(inlist.c_str(), std::ofstream::out | std::ofstream::trunc);
		inp_list << file << std::endl;
		inp_list.close();

		cv::Mat img = cv::imread(infile.c_str(), -1);
		CHECK(!img.empty()) << "Unable to decode image " << file;
		Segmenter Segmenter(model_file, trained_file, label_file);
		cv::Mat out = Segmenter.Segment();

		cv::imwrite(out_dir + "/" + model + "_" + file.substr(0, file.length()-4) + ".png", out);
		time(&timer2);
		printf("segmentation took %.f seconds\n", difftime(timer2, timer1));
	    }
	  }
	}
  }
  (void)inotify_rm_watch(fd, wd[0]);
  (void)close(fd);
}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
