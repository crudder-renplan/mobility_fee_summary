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

    - versions - these will vary depending the stage of development, the most recent version
        reflects the current state of code

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
        - read in [Infill Area](https://gis-mdc.opendata.arcgis.com/datasets/MDC::urban-infill-area-1998-polygon/about) 
            - accessed geometry from Miami-Dade Open Data Portal
        - read in rif_fee_table (draft fee for each LU with rate per units (sf or otherwise)
        - read in land use tables 
            - MD DOR use codes mapped to LU names and more generalized LU types
        - read in parcels to spatialize the permits
            - MD parcel data accessed MD DOR
    2) For each year:
        - define scaling factor base on year (applied to RIF schedule)
        - read in permit data
        - scale rif_fee_table based on year
        - join mf and rif sched tables to permits based on LU
        - read in land use tables to permit data based on LU code in permits
        - clean up files and rename columns
            - generate points from parcels and join permits to points with same FOLIO
        - generate a spatial dataset by utilizing the parcel centroids to geocode the permits
            - dump out a shapefile for validation
            - intersect with infill to identify parcel inside and outside infill area
            - intersect with Context Areas to acquire context area attributes (SMART Corridor, and Ring)
        # CALCULATE RIF and MF from schedule and units
        - generate calculated Road Impact Fee from rif schedule table and # of units permit has assigned
        - generate calculated Mobility Fee from mf schedule table and # of units permit has assigned based on Corridor/RING combo
        # CALCULATE Actual RIF in DB (for permits with multiple rows [ie. credits for previous land uses and new uses], 
            the Actual RIF value is listed for each record as the same value, to adjust each row and back 
            into an estimate of the actual fees/credits per row the below process is used:
        - generate an adjustment factor (const_fee + admin_fee + credits) / sum(calculated rif fee by landuse)
        - calculate Road Impact fee (Actual): RIF_CALCULATED * adjustment factor
        # AGGREGATE
        - Aggregate data on:
            ["Ring", "SMART", "PED_ORIENTED", "PROC_NUM", "CAT_CODE", "LANDUSE", "SPC_LU", "GN_VA_LU", "UNITS", ]
            - sum "RIF_CALCULATED", "MF_CALCULATED", "RIF_ACTUAL" for the aggregations
        # DIFFERENCE RIF_calc/MF_calc/RIF_actual
        - calculate the difference between the fee estimates and db
            - "DIFF_RFc_RFa", "DIFF_MFc_RFc", "DIFF_MFc_RFa"
        - add a year attribute
        - fix and lingering oddities in the table (land use codes)
        - write to CSV
    3) combine all years into a single file and write out to CSV``