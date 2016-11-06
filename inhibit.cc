#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <tuple>
#include <queue>
#include <unordered_set>
#include <math.h>

using namespace std;
using namespace tensorflow;

REGISTER_OP("Inhibit")
    .Input("to_inhibit: float")
    .Output("inhibited: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

/**
 * Applies inhibition to a matrix, setting all but the highest values
 * in the matrix to zero, and all highest values to one.
 */
class InhibitOp : public OpKernel {
 public:
  explicit InhibitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    const int maxActive = (int) ceil(N * 0.2f);

    priority_queue<tuple<int, int>> keep;

    for (int i = 0; i < N; i++) {
      auto score = input(i);

      if(keep.empty() || score > get<0>(keep.top())) {
        // We want to keep this
        if(keep.size() >= maxActive) {
          // Replace lowest score with this one
          keep.pop();
        }

        keep.push(make_tuple(score, i));
      }
    }

    // Create a set of indicies to keep
    unordered_set<int> keepIndicies;

    while(!keep.empty()) {
      keepIndicies.insert(get<1>(keep.top()));
      keep.pop();
    }

    // Handle output
    for (int i = 0; i < N; i++) {
      if(keepIndicies.find(i) != keepIndicies.end())
        output(i) = 1;
      else
        output(i) = 0;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Inhibit").Device(DEVICE_CPU), InhibitOp);
