// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/label_context_aggregator_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {


/* n: number of threads

*/
template <typename Dtype>
__global__ void MatrixUnique(const int map_size, int n, int height, int width, const int num_classes,
          const int xs, const int ys, const int xe, const int ye,
          const int i, const int j, const int H_m, const int W_m,  const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, map_size) {
    int W_f = xe-xs;
    int h = ys + index / W_f;
    int w = xs + index % W_f;
    int c = in[n * height * width + h * width + w];
    int pos = n * num_classes * H_m * W_m + c * H_m * W_m + i * W_m + j;
    if (c < 255){
      if (out[pos] == 0){
        int old = caffe_gpu_atomic_add( Dtype(1.0), out + pos );
      }
    }
  }
}

template <typename Dtype>
__global__ void MatrixReLU(const int n, Dtype* in) {
  CUDA_KERNEL_LOOP(index, n) {
    *(in + index) = *(in + index) > 0 ? Dtype(1.0) : Dtype(0.0);
  }
}

/*
  Layer input:
    bottom[0]: semantic labels for whole image
    bottom[1]: feature map
*/
template <typename Dtype>
void LabelContextAggregatorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // label context aggregator forward, layer params set up:
  caffe_gpu_set(top[0]->count(), Dtype(0.), top[0]->mutable_gpu_data());
  int up_window = down_factor_ * window_;
  int up_pad = down_factor_ * pad_;
  Dtype *top_data = top[0]->mutable_gpu_data();
  const Dtype *bottom_data = bottom[0]->gpu_data();

  // Start calculating label context:
  // loop over N images
  for (int n = 0; n < N_; n++){
    for (int i = 0; i < H_; i++){
      for (int j = 0; j < W_; j++){
        int xs, ys, xe, ye;
        xs = (j * down_factor_ - up_pad) > 0 ? (j * down_factor_ - up_pad) : 0;
        ys = (i * down_factor_ - up_pad) > 0 ? (i * down_factor_ - up_pad) : 0;
        xe = (xs + up_window) < bottom[0]->shape(3) ? (xs + up_window) : bottom[0]->shape(3);
        ye = (ys + up_window) < bottom[0]->shape(2) ? (ys + up_window) : bottom[0]->shape(2);
        int map_size = (xe - xs) * (ye - ys);
        MatrixUnique<Dtype><<<CAFFE_GET_BLOCKS(map_size), CAFFE_CUDA_NUM_THREADS>>>(map_size, n, bottom[0]->shape(2),
                bottom[0]->shape(3), num_classes_, xs, ys, xe, ye, i, j, H_, W_, bottom_data + bottom[0]->offset(n), top_data + top[0]->offset(n));
        CUDA_POST_KERNEL_CHECK;
      }
    }
  }
  MatrixReLU<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(top[0]->count(), top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void LabelContextAggregatorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  ;
  /*if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TanHBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }*/
}

INSTANTIATE_LAYER_GPU_FUNCS(LabelContextAggregatorLayer);


}  // namespace caffe
