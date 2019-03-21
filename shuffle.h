#ifndef __RANDOM_SHUFFLE
#define __RANDOM_SHUFFLE

#include <stdlib.h>

// A random_shuffle-like function that uses the re-entrant random number generator
template <class T>
void randomize(const T &m_begin, const T &m_end, drand48_data *m_ptr_seed)
{
	const size_t len = m_end - m_begin;
	
	if(m_ptr_seed == NULL){
		throw __FILE__ ":randomize: m_ptr_seed == NULL";
	}
	
	for(size_t i = 0;i < len;++i){
	
		double r;
		
		if(drand48_r(m_ptr_seed, &r) != 0){
			throw __FILE__ ":randomize: Error in random_r";
		}
		
		// Generate a random number between [0, len)
		size_t index = size_t( r*len );
		
		while(index == len){
			
			if(drand48_r(m_ptr_seed, &r) != 0){
				throw __FILE__ ":randomize: Error in random_r";
			}
		
			index = size_t( r*len );
		}
		
		std::swap( *(m_begin + i), *(m_begin + index) );
	}
}

#endif // __RANDOM_SHUFFLE
