// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/category_filter_layer.hpp"

namespace caffe {

template <typename Dtype>
void CategoryFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = bottom[0]->shape(0);
}

template <typename Dtype>
void CategoryFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom 0 : semantic label image
  // bottom 1 : feature map input
  N_ = bottom[0]->shape(0);

  top[0]->ReshapeLike(*bottom[0]); // (N, C) - > N for batchsize,  C for number of classes, H and W for feature map size
}


template <typename Dtype>
void CategoryFilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void CategoryFilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(CategoryFilterLayer);
#endif

//INSTANTIATE_CLASS(CategoryFilterLayer);
INSTANTIATE_CLASS(CategoryFilterLayer);
REGISTER_LAYER_CLASS(CategoryFilter);

}  // namespace caffe
