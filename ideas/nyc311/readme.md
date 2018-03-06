##          A Machine Learning Project Proposal
####               by Cedric Bhihe, 2018.03.02


***Urban complaint calls to 311 in NYC:***

Each year, starting in 2011, between 70,000 and 120,000 calls to 311 are received by NYC's 
public office and logged with a slew of metadata attributes. Calls are registered with 50+ 
attributes, describing time of the year, gender of caller, exact geolocation pertaining to 
the complaint, etc.


Year of availability: Data is available in annual batches from 2011 to 2015 (both included). 
Data is not included here for your perusal because of upload file size limitation on GitHub. 
It is however readily available at:
https://nycopendata.socrata.com/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9


Data usage: All the data is public, regulated by the terms of use of data and information 
available on the web page: http://www1.nyc.gov.  The terms and conditions of use are included 
here and available at http://www1.nyc.gov/home/terms-of-use.page.


Data is available either for download or via the Socrata Open Data API (SODA), SODA provides 
programmatic access to the datasets including the ability to filter, query, and aggregate data.  
Data comes as csv, json o geojson files (one file per year) weighing about 200MB.  The API is 
well documented. All communication with the API is done through HTTPS, and errors are 
communicated through HTTP response codes. Available response types include JSON, XML, and CSV, 
which are selectable by the "extension" (.json, etc.) on the API endpoint or through content
-negotiation with HTTP Accepts headers.  The documentation also includes inline, runable examples.


Every dataset entry is labeled.  A rapid inspection shows that "NA" (non-assigned) values exist but in small proportion, making it possible to carry out routine imputation or suppression without incurring in issues of bias or unreliability.


In very broad, non-exhaustive terms, it would be interesting to study:
- Clasification
- Clustering of like- or similar complaints
- Geo-correlative study (GPS data for each complaint is available) for instance with SAT scores 
to explore the relationship between quantity, type and variety  of complpaints and educational 
level or wealth.
- Density estimation (mechanisms which generate the data)
- Predictibility of complaints, e.g.:
	- use 2013 as training set and estimate trends in 2014
	- detect micro trends that announce correlated complaints, taking seasonality into account 
	and wealth level of district, where the compaint is registered.

In addition, although not necessary for clustering, categorical data can be easily transformed 
(pre-processing) in times series over sliding windows of varying widths.
