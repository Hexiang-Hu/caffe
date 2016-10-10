// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/category_filter_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void FilterForward(const int n, Dtype map_size, Dtype* map, const Dtype* filt_map) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = index/map_size;
    *(map + index) = (*(map + index)) * (*(filt_map + c));
  }
}

template <typename Dtype>
__global__ void FilterBackward(const int n, Dtype map_size, const Dtype* top_diff, Dtype* map_diff, Dtype* filt_diff, const Dtype* map, const Dtype* filt) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = index/map_size;
    // map diffs:
    float old = caffe_gpu_atomic_add((*(top_diff + index)) * (*(filt + c)), map_diff + index);
    // filt diffs
    old = caffe_gpu_atomic_add((*(top_diff + index)) * (*(map + index)), filt_diff + c);
  }
}

/*
  Layer input: 
    bottom[0]: score map
    bottom[1]: filter map
*/
template <typename Dtype>
void CategoryFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // label context aggregator forward, layer params set up:
  Dtype *top_data = top[0]->mutable_gpu_data();
  const Dtype *filt_data   = bottom[1]->gpu_data();
  
  caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top_data);
  // Start calculating label context:
  // loop over N images
  int C = bottom[0]->shape(1);
  int single_count = bottom[0]->count()/N_;
  int map_size = single_count/C;
  for (int n = 0; n < N_; n++){
     FilterForward<Dtype><<<CAFFE_GET_BLOCKS(single_count), CAFFE_CUDA_NUM_THREADS>>>(single_count, map_size,
                         top_data + top[0]->offset(n), filt_data + bottom[1]->offset(n));
     CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void CategoryFilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0] && propagate_down[1]) {
    Dtype* map_diff  = bottom[0]->mutable_gpu_diff();
    Dtype* filt_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* map_data = bottom[0]->gpu_data();
    const Dtype* filt_data = bottom[1]->gpu_data();
   
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), map_diff);
    caffe_gpu_set(bottom[1]->count(), Dtype(0.0), filt_diff); 
    int C = bottom[0]->shape(1);
    int single_count = bottom[0]->count()/N_;
    int map_size = single_count/C;
    for (int n = 0; n < N_; n++){    
      FilterBackward<Dtype><<<CAFFE_GET_BLOCKS(single_count), CAFFE_CUDA_NUM_THREADS>>>(
          single_count, map_size, top_diff + top[0]->offset(n), map_diff + bottom[0]->offset(n),
          filt_diff + bottom[1]->offset(n), map_data + bottom[0]->offset(n), filt_data + bottom[1]->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }
  }else{
    std::cout << "gradient not propagated properly to two bottoms!" << std::endl;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CategoryFilterLayer);

}  // namespace caffe
