#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalized_sigmoid_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NormalizedSigmoidCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NormalizedSigmoidCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_indicators_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    // Fill the indicator vector
    for (int i = 0; i < blob_bottom_indicators_->count(); ++i) { blob_bottom_indicators_->mutable_cpu_data()[i] = int( caffe_rng_rand() % 2 ) ; }
    blob_bottom_vec_.push_back(blob_bottom_indicators_);
    // Setup top vector
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~NormalizedSigmoidCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_bottom_indicators_;
    delete blob_top_loss_;
  }

  Dtype SigmoidCrossEntropyLossReference(const int count, const int num, const int factor,
                                         const int width, const int height,
                                         const Dtype* input,
                                         const Dtype* indicator,
                                         const Dtype* target) {
    Dtype loss = 0;
    for (int i = 0; i < count; ++i) {
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      int n = (int) i / factor;
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], 0);
      EXPECT_LE(indicator[n], 1);
      EXPECT_GE(indicator[n], -1);
      loss -= indicator[n] * ( target[i] * log(prediction + (target[i] == Dtype(0)))
                + (1 - target[i]) * log(1 - prediction + (target[i] == Dtype(1))) );
    }
    return loss / (  num * width * height  );
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0.0);
    targets_filler_param.set_max(1.0);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    Dtype eps = 2e-2;
    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
      // Fill the indicators vector
      for (int _i = 0; _i < this->blob_bottom_indicators_->count(); ++_i) {
        this->blob_bottom_indicators_->mutable_cpu_data()[_i] = int( caffe_rng_rand() % 2 );
        EXPECT_LE(this->blob_bottom_indicators_->cpu_data()[_i],  1);
        EXPECT_GE(this->blob_bottom_indicators_->cpu_data()[_i],  0);
      }

      NormalizedSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const int width  = this->blob_bottom_data_->shape(2);
      const int height = this->blob_bottom_data_->shape(3);
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets = this->blob_bottom_targets_->cpu_data();
      const Dtype* blob_bottom_indicators = this->blob_bottom_indicators_->cpu_data();
      Dtype reference_loss = kLossWeight * SigmoidCrossEntropyLossReference(count, num, 5,
          width, height, blob_bottom_data, blob_bottom_indicators,  blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_bottom_indicators_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizedSigmoidCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizedSigmoidCrossEntropyLossLayerTest, TestSigmoidCrossEntropyLoss) {
  this->TestForward();
}

TYPED_TEST(NormalizedSigmoidCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  NormalizedSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
