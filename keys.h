#ifndef __KEYS
#define __KEYS

#include <deque>
#include <unordered_map>
#include "deque_set.h"

template <class K, class V>
std::deque<K> keys(const std::unordered_multimap<K, V> &m_data)
{
	std::deque<K> ret;
	
	for(typename std::unordered_multimap<K, V>::const_iterator i = m_data.begin();i != m_data.end();++i){
		ret.push_back(i->first);
	}
	
	make_set(ret);
	
	return ret;
}

template <class K, class V>
std::deque<K> keys(const std::unordered_map<K, V> &m_data)
{
	std::deque<K> ret;
	
	for(typename std::unordered_map<K, V>::const_iterator i = m_data.begin();i != m_data.end();++i){
		ret.push_back(i->first);
	}
	
	make_set(ret);
	
	return ret;
}

#endif // __KEYS
