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

    PROCEDURE:
    1) read in data
        - read in Context Areas
        - read in Infill Area
        - read in mobility_fee_table (draft fee for each LU with rate per units (sf or otherwise)
        - read in land use tables
        - read in parcels to spatialize the permits
    2) For each year:
        - read in permit data
        - clean up files and rename columns
        - join mobility_fee_table and land use tables to permit data based on LU code in permits
        - generate points from parcels and join permits to points with same FOLIO
        - use spatial data frame 
            - dump out a shapefile for validation
            - intersect with infill to identify parcel inside and outside infill area
            - intersect with Context Areas to acquire context area attributes (SMART Corridor, and Ring)
        - calculate Mobility Fee from mobility fee rate table and # of units permit has assigned (MF_RT_BASED_FEE)
        - zero out duplicated RIF fee values (Construction, Admin, Credits)
        - calculate Road Impact fee by summing the (Construction, Admin, Credits) fees (ROAD_IMPACT_FEE)
        - Aggregate data on:
            ["Ring", "SMART", "PED_ORIENTED", "CAT_CODE", "LANDUSE", "SPC_LU", "GN_VA_LU", "UNITS", ]
            - sum ROAD_IMPACT_FEE and MF_RT_BASED_FEE for the aggregations
        - calculate the difference between the two fee estimates
        - add a year attribute
        - fix and lingering oddities in the table (land use codes)
        - write to CSV
    3) combine all years into a single file and write out to CSV