#include "xTx.h"
#include "deque_set.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits.h>

using namespace std;

// Global variables for MPI
extern int mpi_numtasks;
extern int mpi_rank;

#define	COLUMN_NOT_FOUND	0xFFFFFFFFFFFFFFFF

vector<string> split(const string &m_buffer, const char &m_delim);
vector<string> split(const char *m_buffer, const char &m_delim);
size_t get_col(const vector<string> &m_header, const string &m_key);
unsigned int str_to_uint(const string &m_str);
unsigned int str_to_int(const string &m_str);
float str_to_float(const string &m_str);
string toupper(const string &m_str);

void parse_dose_response(const string &m_filename, 
	vector< MAP<DrugID, DoseResponse> > &m_data,
	MAP< string, SET< pair<CellID, DrugID> > > &m_source,
	const MAP<string, CellID> &m_cell_names,
	const MAP<string, DrugID> &m_drug_names)
{
	const size_t num_cell = m_cell_names.size();
	
	// The dose response data is indexed [cell][drug]
	// Resize the vector to account for all of the cells
	m_data.resize(num_cell);
	
	ifstream fin( m_filename.c_str() );
	
	if(!fin){
		throw __FILE__ ":parse_dose_response: Unable to open file for reading";
	}
	
	string line;
	
	if( !getline(fin, line) ){
		throw __FILE__ ":parse_dose_response: Unable to read header";
	}
	
	const vector<string> header = split(line, '\t');
	
	const size_t src_col = get_col(header, "SOURCE");
	
	if(src_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"SOURCE\"";
	}

	const size_t drug_id_col = get_col(header, "DRUG_ID");
	
	if(drug_id_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"DRUG_ID\"";
	}
	
	const size_t cellname_col = get_col(header, "CELLNAME");
	
	if(cellname_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"CELLNAME\"";
	}
	
	const size_t concunit_col = get_col(header, "CONCUNIT");
	
	if(concunit_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"CONCUNIT\"";
	}
	
	const size_t log_conc_col = get_col(header, "LOG_CONCENTRATION");
	
	if(log_conc_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"LOG_CONCENTRATION\"";
	}
	
	const size_t expid_col = get_col(header, "EXPID");
	
	if(expid_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"EXPID\"";
	}
	
	const size_t growth_col = get_col(header, "GROWTH");
	
	if(growth_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_dose_response: Unable to find column \"GROWTH\"";
	}
	
	const size_t num_col = header.size();
	size_t line_number = 0;
	
	while( getline(fin, line) ){
		
		++line_number;
		
		// Each rank parses a unique subset of lines
		if( int(line_number%mpi_numtasks) != mpi_rank ){
			continue;
		}
				
		const vector<string> data = split(line, '\t');
		
		if( num_col != data.size() ){
			throw __FILE__ ":parse_dose_response: Did not read the expected number of columns";
		}
		
		// Lookup the id of the cell line
		MAP<string, CellID>::const_iterator cell_iter = m_cell_names.find(data[cellname_col]);
		
		if( cell_iter == m_cell_names.end() ){
			continue;
		}
		
		// Lookup the id of the drug
		MAP<string, DrugID>::const_iterator drug_iter = m_drug_names.find(data[drug_id_col]);
		
		if( drug_iter == m_drug_names.end() ){
			continue;
		}
		
		// Make sure that all of the concentration units are in 'M'
		if(data[concunit_col] != "M"){
			throw __FILE__ ":parse_dose_response: Found a non-M concentration!";
		}
		
		const float growth = str_to_float(data[growth_col]);
		
		// A small sanity check on the growth data (since it should all be clamped
		// to: -100 <= growth <= 300).
		if( (growth < -100.0) || (growth > 300.0) ){
			throw __FILE__ ":parse_dose_response: Growth is out of bounds!";
		}
		
		const float conc = str_to_float(data[log_conc_col]);
		
		// A sanity check on the dose measurement -- this test currently fails due to
		// some outlier values in the NCI-60 dose response dataset
		//if( (conc < -18.0) || (conc > -0.5) ){
		//	throw __FILE__ ":parse_dose_response: conc is out of bounds!";
		//}
		
		if(cell_iter->second >= num_cell){
			throw __FILE__ ":parse_dose_response: Cell index out of bounds";
		}

		m_data[cell_iter->second][drug_iter->second].push_back( make_pair(conc, growth) );
		m_source[ data[src_col] ].insert( make_pair(cell_iter->second, drug_iter->second) );
	}
}

void parse_doubling_time(const string &m_filename, 
	vector< MAP<DrugID, float> > &m_data,
	MAP< string, SET< pair<CellID, DrugID> > > &m_source,
	const MAP<string, CellID> &m_cell_names,
	const MAP<string, DrugID> &m_drug_names,
	MULTIMAP<string, CellID> &m_prefix_to_cell)
{
	if( m_filename.empty() ){
		return;
	}
	
	const char delim = ',';
	
	const size_t num_cell = m_cell_names.size();
	
	// The PDM doubling time data is indexed [cell][drug]
	// Resize the vector to account for all of the cells
	m_data.resize(num_cell);
	
	ifstream fin( m_filename.c_str() );
	
	if(!fin){
		throw __FILE__ ":parse_doubling_time: Unable to open file for reading";
	}
	
	string line;
	
	if( !getline(fin, line) ){
		throw __FILE__ ":parse_doubling_time: Unable to read header";
	}
	
	const vector<string> header = split(line, delim);
	
	const size_t pdx_col = get_col(header, "PDX-Specimen");
	
	if(pdx_col == COLUMN_NOT_FOUND){
		throw __FILE__ ":parse_doubling_time: Unable to find column \"PDX-Specimen\"";
	}
	
	const size_t num_col = header.size();
	size_t line_number = 0;
	
	while( getline(fin, line) ){
		
		++line_number;
		
		// Each rank parses the **entire** file
		//if( int(line_number%mpi_numtasks) != mpi_rank ){
		//	continue;
		//}
				
		const vector<string> data = split(line, delim);
		
		if( num_col != data.size() ){
			throw __FILE__ ":parse_doubling_time: Did not read the expected number of columns";
		}
		
		// The PDX-Specimen names are just name prefixes -- find all cell names that match
		// the given prefix
		const string &name_prefx = data[pdx_col];
		
		// Associate this prefix with the actual gene expression data sets
		// that contain the prefix. Only update this mapping the first time
		// we encounter a given prefix.
		if( m_prefix_to_cell.find(name_prefx) == m_prefix_to_cell.end() ){
		
			for(MAP<string, CellID>::const_iterator cell_iter = m_cell_names.begin();
					cell_iter != m_cell_names.end();++cell_iter){

				// Is name_prefx a prefix of the current cell name?
				if(cell_iter->first.find(name_prefx) != 0){
					continue;
				}

				// Store the prefix to cell id to enable permutation testing
				m_prefix_to_cell.insert( make_pair(name_prefx, cell_iter->second) );
			}
		}
		
		// For each column (i.e. drug)
		for(size_t i = 0;i < num_col;++i){
			
			// Skip the PDX-specimen column
			if(i == pdx_col){
				continue;
			}
			
			// Assume this column is a drug name and look up the drug ID
			MAP<string, DrugID>::const_iterator drug_iter = m_drug_names.find(header[i]);

			if( drug_iter == m_drug_names.end() ){
				continue;
			}
			
			// Skip the PDM/drug combinations with missing data
			const string val = toupper( data[i] );
			
			if( (val == "") || (val == "NA") || (val == "-") ){
				continue;
			}
			
			const float doubling_time = str_to_float(val);
			
			// A small sanity check on the doubling time data
			if( (doubling_time < 0.0) || (doubling_time > 50.0) ){
				throw __FILE__ ":parse_doubling_time: Doubling time is out of bounds!";
			}
			
			// Lookup the id of the cell line
			for(MAP<string, CellID>::const_iterator cell_iter = m_cell_names.begin();
				cell_iter != m_cell_names.end();++cell_iter){
				
				// Is name_prefx a prefix of the current cell name?
				if(cell_iter->first.find(name_prefx) != 0){
					continue;
				}
				
				// We have a valid prefix, so save the doubling time data
				m_data[cell_iter->second][drug_iter->second] = doubling_time;
				
				// Use "NCIPDM" as a hard-coded source
				m_source["NCIPDM"].insert( make_pair(cell_iter->second, drug_iter->second) );
			}
			
		}
	}
}

void parse_feature(const string &m_filename, MAP< string, vector<float> > &m_features, 
	vector<string> &m_feature_names)
{
	ifstream fin( m_filename.c_str() );
	
	if(!fin){
		throw __FILE__ ":parse_feature: Unable to open file for reading";
	}
	
	string line;
	
	if( !getline(fin, line) ){
		throw __FILE__ ":parse_feature: Unable to read header";
	}
	
	const vector<string> header = split(line, '\t');
	
	size_t num_feature = 0;
	size_t num_col = 0;
	
	// The ECFP or PFP fingerprint files do *not* have descriptive, multicolumn headers
	// and we will create a fake header for these feature types later
	if(header.size() > 1){

		num_col = header.size();
		num_feature = num_col - 1;
		
		m_feature_names.assign( header.begin() + 1, header.end() );
	}
	
	size_t line_number =  0;
	
	while( getline(fin, line) ){
		
		++line_number;
		
		// Each rank parses a unique subset of lines
		if( int(line_number%mpi_numtasks) != mpi_rank ){
			continue;
		}
		
		const vector<string> data = split(line, '\t');
		
		if( m_feature_names.empty() ){
			
			// Create a dummy vector of feature names
			num_col = data.size();
			
			if(num_col <= 1){
				throw __FILE__ ":parse_feature: Did not read an expected number of columns";
			}
			
			num_feature = num_col - 1;
			
			m_feature_names.resize(num_feature);
			
			for(size_t i = 0;i < num_feature;++i){
				
				stringstream ssin;
				
				ssin << "Id" << i;
				
				m_feature_names[i] = ssin.str();
			}
		}
		
		if( num_col != data.size() ){
			throw __FILE__ ":parse_dose_response: Did not read the expected number of columns";
		}
		
		const string &id = data[0];
		
		if( m_features.find(id) != m_features.end() ){
		
			cerr << "duplicate Id = " << id << endl;
			throw __FILE__ ":parse_feature: Found a duplicate feature id";
		}
		
		vector<float> &f = m_features[id];
		
		f.resize(num_feature);
		
		for(size_t i = 0;i < num_feature;++i){
			
			const string val = toupper( data[i + 1] );
			
			if(val == "NA"){
				f[i] = MISSING_DATA;
			}
			else if(val == "-"){
				f[i] = MISSING_DATA;
			}
			else if(val == ""){
				f[i] = MISSING_DATA;
			}
			else{
				f[i] = str_to_float(val);
			}
		}
	}
}

vector<string> split(const string &m_line, const char &m_delim)
{
	// Delimiter are litteral if they are contained withing matching protect characters
	const char protect = '"';
	
	vector<string> ret;
	
	const size_t len = m_line.size();
	
	size_t protect_count = 0;
	
	size_t begin = 0;
	size_t end;
	
	for(end = 0;end < len;++end){
		
		// A DOS/Windows '\r' symbol forces the end of the line
		if(m_line[end] == '\r'){
			break;
		}
		
		protect_count += (m_line[end] == protect);
		
		if( (m_line[end] == m_delim) && (protect_count%2 == 0) ){
			
			ret.push_back( m_line.substr(begin, end - begin) );
			begin = end + 1; // Skip the delimeter
		}
	}
	
	if(begin < end){
		ret.push_back( m_line.substr(begin, end - begin) );
	}
	else{
		// Special case of a line that ends with a delimeter
		ret.push_back( string() );
	}

	return ret;
}

size_t get_col(const vector<string> &m_header, const string &m_key)
{
	vector<string>::const_iterator iter = find(m_header.begin(), m_header.end(), m_key);

	if( ( iter == m_header.end() ) || (*iter != m_key) ){
		
		//cerr << "Unable to find the requested column header: " << m_key << endl;
		//throw __FILE__ ":get_col: Unable to find column key";
		return COLUMN_NOT_FOUND;
	}
	
	return ( iter - m_header.begin() );
}

unsigned int str_to_uint(const string &m_str)
{
	size_t ret = 0;
	
	for(string::const_iterator i = m_str.begin();i != m_str.end();++i){
		
		if( !isdigit(*i) ){
			
			cerr << "Error parsing: " << m_str << endl;
			throw __FILE__ ":str_to_uint: Illegal character detected";
		}
		
		ret = ret*10 + (*i - '0');
	}
	
	if(ret > 0xFFFFFFFF){
		throw __FILE__ ":str_to_uint: Overflow";
	}
	
	return (unsigned int)ret;
}

unsigned int str_to_int(const string &m_str)
{
	long long int ret = 0;
	int sign = 1;
	
	for(string::const_iterator i = m_str.begin();i != m_str.end();++i){
		
		if(*i == '-'){
		
			sign = -1*sign;
			continue;
		}
		
		if( !isdigit(*i) ){
			
			cerr << "Error parsing: " << m_str << endl;
			throw __FILE__ ":str_to_uint: Illegal character detected";
		}
		
		ret = ret*10 + (*i - '0');
	}
	
	if(ret > INT_MAX){
		throw __FILE__ ":str_to_int: Overflow";
	}
	
	return (int)(sign*ret);
}

float str_to_float(const string &m_str)
{
	return atof( m_str.c_str() );
}

string toupper(const string &m_str)
{
	string ret(m_str);
	
	for(string::iterator i = ret.begin();i != ret.end();++i){
		*i = ::toupper(*i);
	}
	
	return ret;
}

