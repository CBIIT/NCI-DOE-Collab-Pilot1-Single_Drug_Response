#include <math.h>
#include <vector>
#include <algorithm>


template <class PAIR_CONTAINER>
float pearson_correlation(const PAIR_CONTAINER &m_data)
{
	if( m_data.empty() ){
		throw __FILE__ ":pearson_correlation: No data found!";
	}
	
	const size_t len = m_data.size();
	
	// Accumulate in double
	double ave_x = 0.0;
	double ave_y = 0.0;
	
	for(typename PAIR_CONTAINER::const_iterator i = m_data.begin();i != m_data.end();++i){
		
		ave_x += i->first;
		ave_y += i->second;
	}
	
	ave_x /= len;
	ave_y /= len;
	
	// Accumulate in double
	double xx = 0.0;
	double yy = 0.0;
	double xy = 0.0;
	
	for(typename PAIR_CONTAINER::const_iterator i = m_data.begin();i != m_data.end();++i){
		
		const float dx = (i->first - ave_x);
		const float dy = (i->second - ave_y);
		
		xy += dx*dy;
		xx += dx*dx;
		yy += dy*dy;
	}
	
	if(xy == 0.0){
		return 0.0f;
	}
	
    // DEBUG -- testing the effects of normalizing the variance using (n - 1) instead of (n).
    // Has a small effect ...
    //xy /= len;
    //xx /= (len - 1);
    //yy /= (len - 1);
    
	const float r = xy/sqrt( fabs(xx*yy) );
	
	if( isnan(r) || isinf(r) ){
		
		//cerr << "Invalid pearson correlation r = " << r << endl;
		//cerr << "len = " << len << endl;
		//cerr << "xx = " << xx << endl;
		//cerr << "yy = " << yy << endl;
		//cerr << "xy = " << xy << endl;
		
		//for(typename PAIR_CONTAINER::const_iterator i = m_data.begin();i != m_data.end();++i){
		//	cerr << i->first << '\t' << i->second << endl;
		//}
		throw __FILE__ ":pearson_correlation: Error computing pearson correlation coefficient";
	}
	
	return r;
}

template <class PAIR_CONTAINER>
float spearman_correlation(const PAIR_CONTAINER &m_data)
{
	const unsigned int len = m_data.size();
	
	std::vector< std::pair<float, unsigned int> > local_rank(len);
	std::vector< std::pair<float, float> > final_rank(len);
	
	for(unsigned int i = 0;i < len;++i){
		local_rank[i] = std::make_pair(m_data[i].first, i);		
	}
	
	std::sort( local_rank.begin(), local_rank.end() );
	
	// Allow for ties in the data
	unsigned int index = 0;
	
	while(index < len){
		
		const unsigned int start = index;
	
		do{
			++index;
		}
		while( (index < len) && (local_rank[index].first == local_rank[start].first) );
		
		const float ave_rank = start + 0.5f*( (index - start) - 1 );
		
		for(unsigned int i = start;i < index;++i){
			final_rank[local_rank[i].second].first = ave_rank;
		}
	}

	//////////////////////////////////////////////////////////////////////
	for(unsigned int i = 0;i < len;++i){
		local_rank[i] = std::make_pair(m_data[i].second, i);
	}
	
	std::sort( local_rank.begin(), local_rank.end() );
	
	// Allow for ties in the data
	index = 0;
	
	while(index < len){
		
		const unsigned int start = index;
	
		do{
			++index;
		}
		while( (index < len) && (local_rank[index].first == local_rank[start].first) );
		
		const float ave_rank = start + 0.5f*( (index - start) - 1 );
		
		for(unsigned int i = start;i < index;++i){
			final_rank[local_rank[i].second].second = ave_rank;
		}
	}
	
	return pearson_correlation< std::vector< std::pair<float, float> > >(final_rank);
	
	// The old version below is about two times *slower* that the new version above
	#ifdef OLD_VERSION 
	
	const size_t len = m_data.size();
	
	std::deque< std::pair<float, size_t> > rank_a;
	std::deque< std::pair<float, size_t> > rank_b;
	
	for(size_t i = 0;i < len;++i){
		
		rank_a.push_back( std::make_pair(m_data[i].first, i) );
		rank_b.push_back( std::make_pair(m_data[i].second, i) );
	}
	
	std::sort( rank_a.begin(), rank_a.end() );
	std::sort( rank_b.begin(), rank_b.end() );
	
	// Allow for ties in the data
	std::unordered_map<float, size_t> total_rank_a;
	std::unordered_map<float, size_t> norm_rank_a;
	
	std::unordered_map<float, size_t> total_rank_b;
	std::unordered_map<float, size_t> norm_rank_b;
	
	for(size_t i = 0;i < len;++i){
		
		total_rank_a[rank_a[i].first] += i;
		++norm_rank_a[rank_a[i].first];
		
		total_rank_b[rank_b[i].first] += i;
		++norm_rank_b[rank_b[i].first];
	}
	
	std::deque< std::pair<float, float> > rank(len);
	
	for(size_t i = 0;i < len;++i){
		
		rank[rank_a[i].second].first = total_rank_a[rank_a[i].first]/norm_rank_a[rank_a[i].first];
		rank[rank_b[i].second].second = total_rank_b[rank_b[i].first]/norm_rank_b[rank_b[i].first];
	}
	
	return pearson_correlation< std::deque< std::pair<float,float> > >(rank);
	
	#endif // OLD_VERSION
}
