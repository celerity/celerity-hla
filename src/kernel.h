#ifndef KERNEL_H
#define KERNEL_H

#include "celerity.h"
#include "static_iterator.h"

namespace celerity::algorithm
{
	
template<typename F, typename OutputView, typename InputView>
class transform_kernel
{
public:
	transform_kernel(F f, OutputView output_view, InputView input_view)
		: f_(f), output_view_(output_view), input_view_(input_view) {}

	void operator()(handler cgh) const
	{
		auto output = algorithm::fixed::create_accessor<access_mode::write>(cgh, output_view_);
		auto input = algorithm::fixed::create_accessor<access_mode::read>(cgh, input_view_);

		cgh.parallel_for<class test>(input_view_.range(), [=](auto item)
		{
			auto r = f_(input[item]);
		});
	}

private:
	F f_;
	OutputView output_view_;
	InputView input_view_;
};

template<typename F, typename OutputView, typename InputView>
auto transform(InputView in_view, OutputView out_view, F f)
{
	return transform_kernel<F, OutputView, InputView>{f, out_view, in_view};
}

}
#endif