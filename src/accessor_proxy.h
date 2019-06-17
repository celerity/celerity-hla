#ifndef ACCESSOR_PROXY_H
#define ACCESSOR_PROXY_H

#include "celerity.h"

namespace celerity
{
	template<size_t Rank>
	struct slice
	{
		item<Rank> item;
	};

}

#endif // ACCESSOR_PROXY_H

