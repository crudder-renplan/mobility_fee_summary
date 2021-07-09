# Mobility Fee Summarization comparison (Road Impact Fee)
Project to summarize the Road Impact Fee permits and provide a comparison of those fees with the proposed Mobility Fee

## Whats Included?
1 - docs [folder]
    
    - road_impact_fee_cat_codes.csv - table of categorical land uses by code with units 

    - Road IFS Category Codes.pdf -  original document the above table was created from

    - POD Ordinance.pdf - memo documenting the administrative fee cost

    - Land_Use_Recode.csv - DOR land use codes and their generalized land uses

    - RIF_attribute_metadata.csv - minimal documentation of the columns in the RIF report csv's

2 - outputs [folder]

    - version1 - simple summarization of the source reports long on year, and various land use types

    - version2 - updated summarization of the source reports including a comparison of the RIF vs Mobility Fee schedule

3 - source_reports [folder]

    - reports generated from RER at Miami-Dade County separated by year

4 - MobilityFeeCalculation_v3_chs.xlsb.xlsx [file]

    - spread sheet containing the RIF_Fee_Ratio for each context area, calculated as the ratio of 
        Context Area Fee / Printed RIF

5 - road_impact_fee_to_points.py
    
    - script used to perform summary