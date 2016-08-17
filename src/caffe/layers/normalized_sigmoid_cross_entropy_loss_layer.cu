#include <vector>

#include "caffe/layers/normalized_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void IndicatorBackward(const int n, const int outer_factor, const int inner_factor, const int dim_class, const Dtype* in, const Dtype* indicator, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    int _n = int( index / outer_factor );
    int _m = int( (index % outer_factor) / inner_factor );
    out[index] = in[index] * indicator[_n*dim_class+_m];
  }
}

template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) { LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."; }
  if (propagate_down[1]) { LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."; }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* indicator = bottom[2]->gpu_data();
    const int outer_factor = bottom[0]->shape(1) * bottom[0]->shape(2) * bottom[0]->shape(3);
    const int inner_factor = bottom[0]->shape(2) * bottom[0]->shape(3);
    const int dim_class    = bottom[2]->shape(1);

    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Filter loss use indicator
    IndicatorBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, outer_factor, inner_factor, dim_class, bottom_diff, indicator, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(NormalizedSigmoidCrossEntropyLossLayer);


}  // namespace caffe
