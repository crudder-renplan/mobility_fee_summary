import os
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_float_dtype
from arcgis import GeoAccessor, GeoSeriesAccessor

# utilized to scale 2020 fee schedule to match year
fee_scaler = {2015: 0.85, 2016: 0.88,
              2017: 0.91, 2018: 0.94,
              2019: 0.97, 2020: 1.0}
SCRIPT_PATH = Path(__file__)
PROJ_PATH = SCRIPT_PATH.parent
PERMITS_STATUS_DICT = dict(
    [
        ("C", "Collected"),
        ("A", "Assessed"),
        ("L", "Letter of Credit submitted"),
        ("B", "Bond submitted"),
    ]
)
PERMITS_FIELDS_DICT = dict(
    [
        ("PROC_NUM", "PROC_NUM"),
        ("FOLIO_NUM", "FOLIO"),
        ("SITE_ADDR", "ADDRESS"),
        ("ASSE_DATE", "DATE"),
        ("STATUS_CODE", "STATUS"),
        ("COL_CON", "CONST_COST"),
        ("COL_ADM", "ADMIN_COST"),
        ("CRED_AMT", "CREDITS"),
        ("ASSD_CATTHRES_CATC_CODE", "CAT_CODE"),
        ("ASSD_BASIS_QTY", "UNITS_VAL"),
    ]
)
PERMITS_USE = PERMITS_FIELDS_DICT.keys()
PERMITS_DROPS = ["CONST_COST", "ADMIN_COST"]
PERMITS_CAT_CODE_PEDOR = "PD"


def make_path(in_folder, *subnames):
    """Dynamically set a path (e.g., for iteratively referencing year-specific geodatabases).
        {in_folder}/{subname_1}/../{subname_n}
    Args:
        in_folder (str): String or Path
        subnames (list/tuple): A list of arguments to join in making the full path

    Returns (str):
        str: String path
    """
    return os.path.join(in_folder, *subnames)


def csv_to_df(csv_file, use_cols, rename_dict):
    """
    Helper function to convert CSV file to pandas dataframe, and drop unnecessary columns
    assumes any strings with comma (,) should have those removed and dtypes infered

    Args:
        csv_file (str): path to csv file
        use_cols (list): list of columns to keep from input csv
        rename_dict (dict): dictionary mapping existing column name to standardized column names

    Returns:
        Pandas.DataFrame
    """
    if isinstance(use_cols, str):
        use_cols = [use_cols]
    df = pd.read_csv(filepath_or_buffer=csv_file, usecols=use_cols, thousands=",")
    df = df.convert_dtypes()
    df.rename(columns=rename_dict, inplace=True)
    return df


def add_xy_from_poly(poly_fc, poly_key, table_df, table_key, to_crs=None):
    """
    Calculates x,y coordinates for a given polygon feature class and returns as new
    columns of a data

    Args:
        poly_fc (str): path to polygon feature class
        poly_key (str): primary key from polygon feature class
        table_df (pd.DataFrame): pandas dataframe
        table_key (str): primary key of table df

    Returns:
        pandas.DataFrame: updated `table_df` with XY centroid coordinates appended
    """
    if ".gdb" in poly_fc:
        fds, layer = os.path.split(poly_fc)
        gdb, _ = os.path.split(fds)
        polys = gpd.read_file(filename=gdb, driver="FileGDB", layer=layer)
    else:
        polys = gpd.read_file(poly_fc)
    if to_crs:
        polys.to_crs(epsg=to_crs)
    cols = [col for col in polys.columns.tolist() if col not in ["geometry", poly_key]]
    polys.drop(columns=cols, inplace=True)
    pts = polys.copy()
    pts["geometry"] = pts['geometry'].centroid
    return pd.merge(
        left=table_df, right=pts, how="inner", left_on=table_key, right_on=poly_key,
    )


# permit functions
def clean_permit_data(
        permit_csv,
        parcel_fc,
        permit_key,
        poly_key,
        rif_lu_tbl,
        dor_lu_tbl,
        mobility_fee_df,
        out_file,
        out_crs,
):
    """
    Reformat and clean RER road impact permit data, specific to the TOC tool

    Args:
        permit_csv (str): path to permit csv
        permit_key (str): foreign key of permit data that ties to parcels ("FOLIO")
        rif_lu_tbl (str): path to road_impact_fee_cat_codes table (maps RIF codes to more standard LU codes)
        dor_lu_tbl (str): path to dor use code table (maps DOR LU codes to more standard and generalized categories)
        out_file (str): path to output permit csv

    Returns:
        None
    """
    # TODO: add validation
    # read permit data to dataframe
    permit_df = csv_to_df(
        csv_file=permit_csv,
        use_cols=PERMITS_USE,
        rename_dict=PERMITS_FIELDS_DICT,
    )

    # clean up and concatenate data where appropriate
    #   fix parcelno to string of 13 len
    permit_df[permit_key] = permit_df[permit_key].astype(str)
    permit_df[poly_key] = permit_df[permit_key].apply(lambda x: x.zfill(13))
    if not is_float_dtype(permit_df.CONST_COST):
        permit_df["CONST_COST"] = (
            permit_df["CONST_COST"]
                .apply(lambda x: x.replace("$", ""))
                .apply(lambda x: x.replace(" ", ""))
                .apply(lambda x: x.replace(",", ""))
                .astype(float)
        )
        permit_df["ADMIN_COST"] = (
            permit_df["ADMIN_COST"]
                .apply(lambda x: x.replace("$", ""))
                .apply(lambda x: x.replace(" ", ""))
                .apply(lambda x: x.replace(",", ""))
                .astype(float)
        )
        permit_df["CREDITS"] = (
            permit_df["CREDITS"]
                .apply(lambda x: x.replace("$", ""))
                .apply(lambda x: x.replace(" ", ""))
                .apply(lambda x: x.replace(",", ""))
                .astype(float)
        )
    #   id project as pedestrian oriented
    permit_df["PED_ORIENTED"] = np.where(
        permit_df.CAT_CODE.str.contains(PERMITS_CAT_CODE_PEDOR), 1, 0
    )

    # drop fake data - Keith Richardson of RER informed us that any PROC_NUM/ADDRESS that contains with 'SMPL' or
    #   'SAMPLE' should be ignored as as SAMPLE entry
    ignore_text = [
        "SMPL",
        "SAMPLE",
    ]
    for ignore in ignore_text:
        for col in ["PROC_NUM", "ADDRESS"]:
            permit_df = permit_df[~permit_df[col].str.contains(ignore)]

    # drop records in PROC_NUM not starting with M, N or C
    prefixes = ["M", "N", "C"]
    dfs = []
    for prefix in prefixes:
        dfs.append(permit_df[permit_df.PROC_NUM.str.startswith(prefix)])
    permit_df = pd.concat(dfs)

    # potential filter of records not in Status = Collected
    # permit_df = permit_df[permit_df.STATUS == "C"]

    #   set landuse codes appropriately accounting for ped-oriented dev
    permit_df["CAT_CODE"] = np.where(
        permit_df.CAT_CODE.str.contains(PERMITS_CAT_CODE_PEDOR),
        permit_df.CAT_CODE.str[:-2],
        permit_df.CAT_CODE,
    )
    #   set project status
    permit_df["STATUS"] = permit_df["STATUS"].map(PERMITS_STATUS_DICT, na_action="NONE")
    #   add landuse codes
    lu_df = pd.read_csv(rif_lu_tbl)
    dor_df = pd.read_csv(dor_lu_tbl)
    lu_df = lu_df.merge(right=dor_df, how="inner", on="DOR_UC")
    permit_df = permit_df.merge(right=lu_df, how="inner", on="CAT_CODE")
    permit_df = permit_df.merge(right=mobility_fee_df, how="inner",
                                left_on="CAT_CODE", right_on="CODE")

    # convert to points
    sdf = add_xy_from_poly(
        poly_fc=parcel_fc, poly_key=poly_key, table_df=permit_df, table_key=permit_key, to_crs=out_crs
    )
    sdf.fillna(0.0, inplace=True)

    sdf = gpd.GeoDataFrame(sdf, geometry=sdf['geometry'], crs=out_crs)
    conversion = {col: float for col in sdf.columns if is_numeric_dtype(sdf[col])}
    sdf = sdf.astype(conversion)
    sdf.to_file(out_file)
    return sdf


if __name__ == "__main__":
    # raw data files/paths
    raw_permits = Path(PROJ_PATH, "source_reports")
    RIF_CAT_CODE_TBL = Path(PROJ_PATH, "docs", "road_impact_fee_cat_codes.csv")
    DOR_UC_TBL = Path(PROJ_PATH, "docs", "Land_Use_Recode.csv")
    GEOMS = Path(PROJ_PATH, "geoms")

    # summary data files/paths
    summary_gdf = gpd.read_file(Path(GEOMS, "ContextAreas_Union.shp"))
    summary_gdf = summary_gdf.replace({"SMART": {0: "Out", 1: "In"}})

    # infill polys/data/files
    infill_gdf = gpd.read_file(Path(GEOMS, "UrbanInfillArea.shp"))
    rif_schedule_tbl = Path(PROJ_PATH, "docs", "rif_fee_schedule.csv")

    # parcel data files/paths
    parcel_gdbs = r"C:\OneDrive_RP\OneDrive - Renaissance Planning Group\SHARE\PMT_link\Data\CLEANED"

    # ratio data files/paths
    mobility_fee_data = Path(PROJ_PATH, "docs", "docs/MobilityFeeCalculation_v3_chs.xlsb.xlsx")
    sheet_name = "MF_ByContextArea"
    column_names = [
        "ITE Land Use Code",
        "R1_In",
        "R2_In",
        "R2_Out",
        "R3_In",
        "R3_Out",
        "R4_In",
        "R4_Out",
    ]
    mf_sched_df = pd.read_excel(
        io=mobility_fee_data, sheet_name=sheet_name, header=0, usecols=column_names
    )
    rif_sched_df = pd.read_csv(rif_schedule_tbl)

    out_folder = Path(PROJ_PATH, "outputs", "version4")
    dfs = []
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:
        pdc_mult = fee_scaler[year]
        permits = list(raw_permits.glob(f"*{year}.csv"))
        out_file = permits[0].stem
        parcels = make_path(parcel_gdbs, f"PMT_{year}.gdb", "Polygons", "Parcels")
        if year == 2020:
            parcels = make_path(parcel_gdbs, f"PMT_2019.gdb", "Polygons", "Parcels")

        # scale rif schedule values by year
        rif_df = rif_sched_df.copy()
        rif_df["fee_outside"] *= pdc_mult
        rif_df["fee_inside"] *= pdc_mult

        # scale mf schedule values by year
        mf_df = mf_sched_df * pdc_mult

        # scale
        permit_gdf = clean_permit_data(
            permit_csv=permits[0],
            parcel_fc=parcels,
            permit_key="FOLIO",
            poly_key="FOLIO",
            rif_lu_tbl=RIF_CAT_CODE_TBL,
            dor_lu_tbl=DOR_UC_TBL,
            mobility_fee_df=mf_df,
            out_file=make_path(out_folder, "RIF_Permits_{}.shp".format(year)),
            out_crs=2881,
        )
        permit_gdf.to_crs(crs=summary_gdf.crs, inplace=True)
        permit_gdf = permit_gdf.merge(right=mf_df,
                                      left_on="CAT_CODE", right_on='ITE Land Use Code',
                                      how="inner")

        # id points inside and outside urban infill area
        permit_gdf["infill"] = permit_gdf.intersects(infill_gdf.unary_union)

        # intersect permits with context areas and add RIF/MF comparison columns to get summary data
        intersect = pd.DataFrame(
            gpd.overlay(df1=summary_gdf, df2=permit_gdf, keep_geom_type=False)
        )
        # calculate
        intersect["RIF_CALCULATED"] = np.where(intersect["infill"] == True,
                                               intersect["fee_inside"] * intersect["UNITS_VAL"],
                                               intersect["fee_outside"] * intersect["UNITS_VAL"])
        for ix, row in summary_gdf.iterrows():
            ring = row["Ring"]
            smart = row["SMART"]
            if smart == 1:
                smart = "Within"
            if smart == 0:
                smart = "Outside"
            intersect["MF_CALCULATED"] = np.where(
                intersect["Ring"] == ring & intersect["SMART"] == smart,
                intersect[f"R{ring}_{smart}"] * intersect["UNITS_VAL"],
            )
        intersect["MOBILITY_FEE"] = intersect["COST"] * intersect["RATIO"]
        intersect["DIFF_MF_vs_RF"] = intersect["MOBILITY_FEE"] - intersect["COST"]
        # drop duplicated RIF cons/adm/credit
        # zero out duplicated costs to 0 so the values arent repeated and get sum of Const/Admin/Credit
        intersect.loc[
            intersect.assign(d=intersect.CONST_COST).duplicated(["PROC_NUM", "FOLIO"]),
            "CONST_COST",
        ] = 0.0
        intersect.loc[
            intersect.assign(d=intersect.ADMIN_COST).duplicated(["PROC_NUM", "FOLIO"]),
            "ADMIN_COST",
        ] = 0.0
        intersect.loc[
            intersect.assign(d=intersect.CREDITS).duplicated(["PROC_NUM", "FOLIO"]),
            "CREDITS",
        ] = 0.0
        intersect["ROAD_IMPACT_FEE"] = intersect["CONST_COST"] + intersect["ADMIN_COST"] + intersect["CREDITS"]

        # summarize data by Context Area and various Landuse types
        df = (
            intersect.groupby(["Ring", "SMART", "PED_ORIENTED",
                               "CAT_CODE", "LANDUSE",
                               "SPC_LU", "GN_VA_LU", "UNITS", ]
                              )["ROAD_IMPACT_FEE", "MF_RT_BASED_FEE"].sum().reset_index()
        # "MOBILITY_FEE", "DIFF_MF_vs_RF"
        )
        df["DIFF_MF_vs_RF"] = intersect["MF_RT_BASED_FEE"] - intersect["ROAD_IMPACT_FEE"]
        df["YEAR"] = year
        df["SPC_LU"].replace({"0": "Unknown"}, inplace=True)
        df["GN_VA_LU"].replace({"0": "Unknown"}, inplace=True)
        df.to_csv(path_or_buf=os.path.join(out_folder, f"{out_file}.csv"))
        dfs.append(df)

    # write out summary data
    out_data = pd.concat(dfs)
    out_data.to_csv(os.path.join(out_folder, "RIF_summary_2015-2020.csv"))
