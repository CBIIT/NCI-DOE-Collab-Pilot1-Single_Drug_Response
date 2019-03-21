#ifndef __MPI_UTIL
#define	__MPI_UTIL

#include <mpi.h>
#include <limits.h>
#include <string.h> // For memcpy

#include <deque>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Generic template for simple types. Specialize as needed for more complex types
template<class T>
size_t mpi_size(const T &m_obj)
{
	// Force a *compile* time test of whether this is a native or derived type
	static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
		":mpi_size: Non-fundamental or non-enum type passed as template");
	
	return sizeof(m_obj);
}

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const T &m_obj)
{
	// Force a *compile* time test of whether this is a native or derived type
	static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
		":mpi_pack: Non-fundamental or non-enum type passed as template");
	
	memcpy( m_ptr, &m_obj, sizeof(m_obj) );
	m_ptr += sizeof(m_obj);
	
	return m_ptr;
}

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, T &m_obj)
{
	// Force a *compile* time test of whether this is a native or derived type
	static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
		":mpi_unpack: Non-fundamental or non-enum type passed as template");
	
	memcpy( &m_obj, m_ptr, sizeof(m_obj) );
	m_ptr += sizeof(m_obj);
	
	return m_ptr;
}

////////////////////////////// Specialization for string //////////////////////////////
template<>
size_t mpi_size(const std::string &m_str);

template<>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::string &m_str);

template<>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::string &m_str);

////////////////// Prototypes to the let compiler know what's available /////////////////
// ---> std::pair prototypes
template<class A, class B>
size_t mpi_size(const std::pair<A, B> &m_obj);

template<class A, class B>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::pair<A, B> &m_obj);

template<class A, class B>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::pair<A, B> &m_obj);

// ---> std::deque prototypes
template<class T>
size_t mpi_size(const std::deque<T> &m_obj);

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::deque<T> &m_obj);

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::deque<T> &m_obj);

// ---> std::vector prototypes
template<class T>
size_t mpi_size(const std::vector<T> &m_obj);

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::vector<T> &m_obj);

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::vector<T> &m_obj);

// ---> std::unordered_map prototypes
template<class A, class B>
size_t mpi_size(const std::unordered_map<A,B> &m_obj);

template<class A, class B>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::unordered_map<A,B> &m_obj);

template<class A, class B>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::unordered_map<A,B> &m_obj);

// ---> std::unordered_set prototypes
template<class T>
size_t mpi_size(const std::unordered_set<T> &m_obj);

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::unordered_set<T> &m_obj);

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::unordered_set<T> &m_obj);

// ---> std::unordered_multimap prototypes
template<class A, class B>
size_t mpi_size(const std::unordered_multimap<A, B> &m_obj);

template<class A, class B>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::unordered_multimap<A, B> &m_obj);

template<class A, class B>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::unordered_multimap<A, B> &m_obj);

/////////////////////////////////////////////////////////////////////////////////////////
// Overload for std::pair
/////////////////////////////////////////////////////////////////////////////////////////
template<class A, class B>
size_t mpi_size(const std::pair<A, B> &m_obj)
{
	return mpi_size(m_obj.first) + mpi_size(m_obj.second);
}

template<class A, class B>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::pair<A, B> &m_obj)
{
	m_ptr = mpi_pack(m_ptr, m_obj.first);
	m_ptr = mpi_pack(m_ptr, m_obj.second);
	
	return m_ptr;
}

template<class A, class B>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::pair<A, B> &m_obj)
{
	m_ptr = mpi_unpack(m_ptr, m_obj.first);
	m_ptr = mpi_unpack(m_ptr, m_obj.second);
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Overload for std::deque
/////////////////////////////////////////////////////////////////////////////////////////
template<class T>
size_t mpi_size(const std::deque<T> &m_obj)
{
	size_t len = sizeof(size_t);
	
	for(typename std::deque<T>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		len += mpi_size(*i);
	}
	
	if(len >= INT_MAX){
		throw __FILE__ ":mpi_size::deque<T>: Max size exceeded";
	}
	
	return len;
}

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::deque<T> &m_obj)
{
	size_t len = m_obj.size();
	
	memcpy( m_ptr, &len, sizeof(size_t) );
	m_ptr += sizeof(size_t);
	
	for(typename std::deque<T>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		m_ptr = mpi_pack(m_ptr, *i);
	}
	
	return m_ptr;
}

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::deque<T> &m_obj)
{
	size_t len;
	
	memcpy( &len, m_ptr, sizeof(size_t) );
	m_ptr += sizeof(size_t);
	
	m_obj.resize(len);
	
	for(size_t i = 0;i < len;++i){
		m_ptr = mpi_unpack(m_ptr, m_obj[i]);
	}
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Overload for std::vector<T>
/////////////////////////////////////////////////////////////////////////////////////////
template<class T>
size_t mpi_size(const std::vector<T> &m_obj)
{
	size_t len = sizeof(size_t);
	
	for(typename std::vector<T>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		len += mpi_size(*i);
	}
	
	if(len >= INT_MAX){
		throw __FILE__ ":mpi_size::vector<T>: Max size exceeded";
	}
	
	return len;
}

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::vector<T> &m_obj)
{
	size_t len = m_obj.size();
	
	m_ptr = mpi_pack(m_ptr, len);
	
	for(typename std::vector<T>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		m_ptr = mpi_pack(m_ptr, *i);
	}
	
	return m_ptr;
}

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::vector<T> &m_obj)
{
	size_t len;
	
	m_ptr = mpi_unpack(m_ptr, len);
	
	m_obj.resize(len);
	
	for(size_t i = 0;i < len;++i){
		m_ptr = mpi_unpack(m_ptr, m_obj[i]);
	}
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Overload for std::unordered_map
/////////////////////////////////////////////////////////////////////////////////////////
template<class A, class B>
size_t mpi_size(const std::unordered_map<A,B> &m_obj)
{
	size_t len = sizeof(size_t);
	
	for(typename std::unordered_map<A,B>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		
		len += mpi_size(i->first);
		len += mpi_size(i->second);
	}
	
	if(len >= INT_MAX){
		throw __FILE__ ":mpi_size::unordered_map<A,B>: Max size exceeded";
	}
	
	return len;
}

template<class A, class B>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::unordered_map<A,B> &m_obj)
{
	size_t len = m_obj.size();
	
	m_ptr = mpi_pack(m_ptr, len);
	
	for(typename std::unordered_map<A,B>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		
		m_ptr = mpi_pack(m_ptr, i->first);
		m_ptr = mpi_pack(m_ptr, i->second);
	}
	
	return m_ptr;
}

template<class A, class B>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::unordered_map<A,B> &m_obj)
{
	size_t len;
	
	m_ptr = mpi_unpack(m_ptr, len);
	
	m_obj.clear();
	
	for(size_t i = 0;i < len;++i){
		
		std::pair<A,B> local;
		
		m_ptr = mpi_unpack(m_ptr, local.first);
		m_ptr = mpi_unpack(m_ptr, local.second);
		
		m_obj.insert(local);
	}
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Overload for std::unordered_set
/////////////////////////////////////////////////////////////////////////////////////////
template<class T>
size_t mpi_size(const std::unordered_set<T> &m_obj)
{
	size_t len = sizeof(size_t);
	
	for(typename std::unordered_set<T>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		len += mpi_size(*i);
	}
	
	if(len >= INT_MAX){
		throw __FILE__ ":mpi_size::unordered_set<T>: Max size exceeded";
	}
	
	return len;
}

template<class T>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::unordered_set<T> &m_obj)
{
	size_t len = m_obj.size();
	
	m_ptr = mpi_pack(m_ptr, len);
	
	for(typename std::unordered_set<T>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		m_ptr = mpi_pack(m_ptr, *i);
	}
	
	return m_ptr;
}

template<class T>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::unordered_set<T> &m_obj)
{
	m_obj.clear();
	
	size_t len;
	
	m_ptr = mpi_unpack(m_ptr, len);
	
	for(size_t i = 0;i < len;++i){
		
		T local;
		
		m_ptr = mpi_unpack(m_ptr, local);
		
		m_obj.insert(local);
	}
	
	return m_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Overload for std::unordered_multimap
/////////////////////////////////////////////////////////////////////////////////////////
template<class A, class B>
size_t mpi_size(const std::unordered_multimap<A, B> &m_obj)
{
	size_t len = sizeof(size_t);
	
	for(typename std::unordered_multimap<A, B>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		
		len += mpi_size(i->first);
		len += mpi_size(i->second);
	}
	
	if(len >= INT_MAX){
		throw __FILE__ ":mpi_size::unordered_multimap<A,B>: Max size exceeded";
	}
	
	return len;
}

template<class A, class B>
unsigned char* mpi_pack(unsigned char* m_ptr, const std::unordered_multimap<A, B> &m_obj)
{
	size_t len = m_obj.size();
	
	memcpy( m_ptr, &len, sizeof(size_t) );
	m_ptr += sizeof(size_t);
	
	for(typename std::unordered_multimap<A, B>::const_iterator i = m_obj.begin();i != m_obj.end();++i){
		
		m_ptr = mpi_pack(m_ptr, i->first);
		m_ptr = mpi_pack(m_ptr, i->second);
	}
	
	return m_ptr;
}

template<class A, class B>
unsigned char* mpi_unpack(unsigned char* m_ptr, std::unordered_multimap<A, B> &m_obj)
{
	size_t len;
	
	m_ptr = mpi_unpack(m_ptr, len);
	
	m_obj.clear();
	
	for(size_t i = 0;i < len;++i){
		
		std::pair<A, B> local;
		
		m_ptr = mpi_unpack(m_ptr, local.first);
		m_ptr = mpi_unpack(m_ptr, local.second);
		
		m_obj.insert(local);
	}
	
	return m_ptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Generic broadcast (from the src to all other ranks)
//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void broadcast(T &m_obj, bool m_is_src)
{
	size_t len = m_is_src ? mpi_size(m_obj) : 0;

	if(len >= INT_MAX){
		throw __FILE__ ":broadcast: Max size exceeded";
	}
	
	MPI_Bcast( &len, sizeof(len), MPI_BYTE, 0, MPI_COMM_WORLD );
	
	unsigned char *buffer = new unsigned char[len];
	
	if(buffer == NULL){
		throw __FILE__ ":broadcast: Unable to allocate buffer";
	}
	
	if(m_is_src){
		mpi_pack(buffer, m_obj);		
	}

	MPI_Bcast(buffer, len, MPI_BYTE, 0, MPI_COMM_WORLD );
		
	if(!m_is_src){
		mpi_unpack(buffer, m_obj);
	}
		
	delete [] buffer;
}

#endif // __MPI_UTIL
