#ifndef __DEQUE_SET
#define __DEQUE_SET

#include <deque>
#include <algorithm>

// A way to potentially speed up the sorting routines (which can consume a non-trivial
// amount of CPU time) is to use the g++-specific __gnu_parallel::sort funcion instead
// of the std::sort function. Since __gnu_parallel::sort is a drop in replacement for
// std::sort, we could include:
//#include <parallel/algorithm>
// and define
//#define	SORT	__gnu_parallel::sort
// or
#define	SORT	std::sort
// to switch between the two versions. This approach still needs benchmarking on the target machine
// (for speed and correctness). This approach will not work on non-g++ compilers (i.e. clang or intel).


template <class T>
void make_set(std::deque<T> &m_set)
{
	SORT( m_set.begin(), m_set.end() );
	m_set.erase( unique( m_set.begin(), m_set.end() ), m_set.end() );
}

template <class T>
std::deque<T> intersection(const std::deque<T> &m_A, const std::deque<T> &m_B)
{
	std::deque<T> ret;
	
	// All inputs must be sorted in ascending order and contain unique elements
	typename std::deque<T>::const_iterator a = m_A.begin();
	typename std::deque<T>::const_iterator b = m_B.begin();
	
	while( ( a != m_A.end() ) && ( b != m_B.end() ) ){
		
		if(*a == *b){
		
			ret.push_back(*a);
			++a;
			++b;
		}
		else{
			if(*a < *b){
				++a;
			}
			else{ // *a > *b
				++b;
			}
		}
	}
	
	return ret;
}

template<class T>
bool set_contains(const std::deque<T> &m_set, const T &m_query)
{
	typename std::deque<T>::const_iterator i = std::lower_bound(m_set.begin(), m_set.end(), m_query);
	
	return ( i != m_set.end() ) && (*i == m_query);
}

template<class T>
size_t get_index(const std::deque<T> &m_set, const T &m_query)
{
	typename std::deque<T>::const_iterator iter = 
		std::lower_bound(m_set.begin(), m_set.end(), m_query);
	
	if( ( iter == m_set.end() ) || (*iter != m_query) ){
		throw __FILE__ ":get_index: Unable to find index!";
	}
	
	return ( iter - m_set.begin() );
};

template<class T>
bool operator==(const std::deque<T> &m_a, const std::deque<T> &m_b)
{
	const size_t len = m_a.size();
	
	if( len != m_b.size() ){
		return false;
	}
	
	for(size_t i = 0;i < len;++i){
		if(m_a[i] != m_b[i]){
			return false;
		}
	}
	
	return true;
}

template<class T>
bool operator!=(const std::deque<T> &m_a, const std::deque<T> &m_b)
{
	const size_t len = m_a.size();
	
	if( len != m_b.size() ){
		return true;
	}
	
	for(size_t i = 0;i < len;++i){
		if(m_a[i] != m_b[i]){
			return true;
		}
	}
	
	return false;
}

template <class T> 
bool find_in_set(const std::deque<T> &m_set, const T &m_query)
{
	typename std::deque<T>::const_iterator iter = 
		lower_bound(m_set.begin(), m_set.end(), m_query);
	
	if( ( iter == m_set.end() ) || (*iter != m_query) ){
		return false;
	}
	
	return true;
}

#endif // __DEQUE_SET
