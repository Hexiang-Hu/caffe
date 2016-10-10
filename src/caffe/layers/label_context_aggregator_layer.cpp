// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/label_context_aggregator_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelContextAggregatorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const LabelContextAggregatorParameter& label_aggregator_param = this->layer_param_.label_context_aggregator_param();
  down_factor_ = label_aggregator_param.down_factor();
  window_ = label_aggregator_param.window();
  pad_ = label_aggregator_param.pad();
  num_classes_ = label_aggregator_param.num_classes();
  N_ = bottom[1]->shape(0);
  H_ = bottom[1]->shape(2);  
  W_ = bottom[1]->shape(3);
}

template <typename Dtype>
void LabelContextAggregatorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom 0 : semantic label image
  // bottom 1 : feature map input
  N_ = bottom[1]->shape(0);
  H_ = bottom[1]->shape(2);
  W_ = bottom[1]->shape(3);

  int num_fea_axes = bottom[1]->num_axes();
  std::vector<int> shape(num_fea_axes, 1);
  shape = bottom[1]->shape();
  shape[1] = num_classes_;
  top[0]->Reshape(shape); // (N, C, H, W) - > N for batchsize,  C for number of classes, H and W for feature map size
}


template <typename Dtype>
void LabelContextAggregatorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::cout << "NOT IMPLEMENTED YET." <<std::endl;
  /*
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }*/
}

template <typename Dtype>
void LabelContextAggregatorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  ;/*
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(LabelContextAggregatorLayer);
#endif

INSTANTIATE_CLASS(LabelContextAggregatorLayer);
REGISTER_LAYER_CLASS(LabelContextAggregator);

}  // namespace caffe
