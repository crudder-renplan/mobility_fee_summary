import os
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from arcgis import GeoAccessor, GeoSeriesAccessor

fee_multiplier = {2015: 1.43, 2016: 1.477,
                  2017: 1.527, 2018: 1.577,
                  2019: 1.628, 2020: 1.682}
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
        mobility_fee_tbl,
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
    if permit_df.CONST_COST.dtype != float:
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
    permit_df["RIF_TOTAL"] = permit_df["CONST_COST"] + permit_df["ADMIN_COST"] + permit_df["CREDITS"]
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
    #   set landuse codes appropriately accounting for pedoriented dev
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
    mob_fee_df = pd.read_csv(mobility_fee_tbl)
    # lu_df = lu_df.join(other=dor_df, on="DOR_UC", how="inner", rsuffix="_")
    lu_df = lu_df.merge(right=dor_df, how="inner", on="DOR_UC")
    permit_df = permit_df.merge(right=lu_df, how="inner", on="CAT_CODE")
    permit_df = permit_df.merge(right=mob_fee_df, how="inner",
                                left_on="CAT_CODE", right_on="CODE")
    # #   drop unnecessary columns
    # permit_df.drop(columns=PERMITS_DROPS, inplace=True)

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
    MOBILITY_FEE_TBL = Path(PROJ_PATH, "docs", "draft_mobility_fee.csv")

    # parcel data files/paths
    parcel_gdbs = r"C:\OneDrive_RP\OneDrive - Renaissance Planning Group\SHARE\PMT_link\Data\CLEANED"

    # ratio data files/paths
    ratio_file = Path(PROJ_PATH, "MobilityFeeCalculation_v3_chs.xlsb.xlsx")
    sheet_name = "RIF_FEE_RATIO"
    column_names = [
        "ITE Land Use Code",
        "Land Use",
        "Unit",
        "Printed RIF (Current Fee)",
        "R1_In_RATIO",
        "R2_In_RATIO",
        "R2_Out_RATIO",
        "R3_In_RATIO",
        "R3_Out_RATIO",
        "R4_In_RATIO",
        "R4_Out_RATIO",
    ]
    ratio_df = pd.read_excel(
        io=ratio_file, sheet_name=sheet_name, header=0, usecols=column_names
    )

    out_folder = Path(PROJ_PATH, "outputs", "version2")
    dfs = []
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:
        pdc_mult = fee_multiplier[year]
        permits = list(raw_permits.glob(f"*{year}.csv"))
        out_file = permits[0].stem
        parcels = make_path(parcel_gdbs, f"PMT_{year}.gdb", "Polygons", "Parcels")
        if year == 2020:
            parcels = make_path(parcel_gdbs, f"PMT_2019.gdb", "Polygons", "Parcels")
        permit_gdf = clean_permit_data(
            permit_csv=permits[0],
            parcel_fc=parcels,
            permit_key="FOLIO",
            poly_key="FOLIO",
            rif_lu_tbl=RIF_CAT_CODE_TBL,
            dor_lu_tbl=DOR_UC_TBL,
            mobility_fee_tbl=MOBILITY_FEE_TBL,
            out_file=make_path(out_folder, "RIF_Permits_{}.shp".format(year)),
            out_crs=2881,
        )
        permit_gdf.to_crs(crs=summary_gdf.crs, inplace=True)
        permit_gdf = permit_gdf.merge(right=ratio_df,
                                      left_on="CAT_CODE", right_on='ITE Land Use Code',
                                      how="inner")
        # zero out duplicated costs to 0 so the values arent repeated
        permit_gdf.loc[
            permit_gdf.assign(d=permit_gdf.CONST_COST).duplicated(["PROC_NUM", "FOLIO"]),
            "CONST_COST",
        ] = 0.0
        permit_gdf.loc[
            permit_gdf.assign(d=permit_gdf.ADMIN_COST).duplicated(["PROC_NUM", "FOLIO"]),
            "ADMIN_COST",
        ] = 0.0
        permit_gdf.loc[
            permit_gdf.assign(d=permit_gdf.CREDITS).duplicated(["PROC_NUM", "FOLIO"]),
            "CREDITS",
        ] = 0.0

        # id points inside and outside urban infill area
        permit_gdf["infill"] = permit_gdf.intersects(infill_gdf.unary_union)

        # intersect permits with context areas and add RIF/MF comparison columns to get summary data
        intersect = pd.DataFrame(
            gpd.overlay(df1=summary_gdf, df2=permit_gdf, keep_geom_type=False)
        )
        intersect["CONST_COST"] = intersect["CONST_COST"] * pdc_mult
        intersect["ADMIN_COST"] = intersect["ADMIN_COST"] * pdc_mult
        intersect["CREDITS"] = intersect["CREDITS"] * pdc_mult
        intersect["RIF_TOTAL"] = intersect["CONST_COST"] + intersect["ADMIN_COST"] + intersect["CREDITS"]

        # incorporate
        for ix, row in summary_gdf.iterrows():
            ring = row["Ring"]
            smart = row["SMART"]
            intersect["RATIO"] = np.where(
                (intersect["Ring"] == ring) & (intersect["SMART"] == smart),
                intersect[f"R{ring}_{smart}_RATIO"], 0
            )
        # intersect["MOBILITY_FEE"] = intersect["RIF_TOTAL"] * intersect["RATIO"]
        # intersect["DIFF_MF_vs_RF"] = intersect["MOBILITY_FEE"] - intersect["RIF_TOTAL"]

        agg_funcs = {
            "UNITS_VAL": "sum",
            "CONST_COST": "sum",
            "ADMIN_COST": "sum",
            "CREDITS": "sum",
            "RIF_TOTAL": "sum"
        }
        # summarize data by Context Area and various Landuse types
        df = (
            intersect.groupby(["Ring", "SMART", "PED_ORIENTED",
                               "CAT_CODE", "LANDUSE",
                               "SPC_LU", "GN_VA_LU", "UNITS", ]
                              )["UNITS_VAL", "CONST_COST", "ADMIN_COST",
                                "CREDITS", "RIF_TOTAL"].sum().reset_index()  # "MOBILITY_FEE", "DIFF_MF_vs_RF"
        )
        df["YEAR"] = year
        df["RIF_TOTAL"] = df["RIF_TOTAL"] / df["UNITS_VAL"]
        # df.rename(columns={"SPC_LU":"LANDUSE"}, inplace=True)
        df["SPC_LU"].replace({"0": "Unknown"}, inplace=True)
        df["GN_VA_LU"].replace({"0": "Unknown"}, inplace=True)
        df.to_csv(path_or_buf=os.path.join(out_folder, f"{out_file}.csv"))
        dfs.append(df)

    # write out summary data
    out_data = pd.concat(dfs)
    out_data.to_csv(os.path.join(out_folder, "RIF_summary_2015-2020.csv"))