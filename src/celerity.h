#ifndef CELERITY_H
#define CELERITY_H

template<size_t Rank>
using range = std::array<int, Rank>;

template<size_t Rank, size_t...Is>
int dispatch_count(range<Rank> r, std::index_sequence<Is...>)
{
	return (std::get<Is>(r) * ... * 1);
}

template<size_t Rank>
int count(range<Rank> r)
{
	return dispatch_count(r, std::make_index_sequence<Rank>{});
}

template<size_t Rank>
using item = std::array<int, Rank>;

struct handler
{
	int invocations;

	template<typename T, size_t Rank, typename F>
	void parallel_for(range<Rank> r, F f)
	{
		for (int i = 0; i < count(r); ++i)
		{
			f(r);
		}
	}

};

class distr_queue
{
public:
	template<typename F>
	void submit(F f)
	{
		f(handler{ ++invocation_count_ });
	}

private:
	int invocation_count_ = 0;
};

enum access_mode
{
	read,
	write,
	read_write
};

template<access_mode mode, typename T, size_t Rank>
struct accessor
{
	T operator[](item<Rank>) { return T{}; }
	T operator[](item<Rank>) const { return T{}; }
};

template<typename T, size_t Rank>
struct buffer
{
	template<access_mode mode>
	auto get_access(handler cgh, range<Rank> range) { return accessor<mode, T, Rank>{}; }
};

#endif