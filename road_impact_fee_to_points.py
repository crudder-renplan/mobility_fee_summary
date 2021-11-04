import datetime
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from pandas.api.types import is_numeric_dtype

# utilized to scale 2020 fee schedule to match year
fee_scaler = {
    2015: 1.43,
    2016: 1.477,
    2017: 1.527,
    2018: 1.577,
    2019: 1.628,
    2020: 1.682,
}
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
    assumes any strings with comma (,) should have those removed and dtypes inferred

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
        to_crs (int): epsg code identifying the output crs used in the procedure, if not
            provided no transformation is made

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
    pts["geometry"] = pts["geometry"].centroid
    return pd.merge(
        left=table_df,
        right=pts,
        how="inner",
        left_on=table_key,
        right_on=poly_key,
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
        permit_csv (str; Path): path to permit csv
        parcel_fc (str; Path): path to parcel feature class
        permit_key (str): foreign key of permit data that ties to parcels ("FOLIO")
        poly_key (str): primary key identifying unique parcel ("FOLIO")
        rif_lu_tbl (str; Path): path to road_impact_fee_cat_codes table (maps RIF codes to more standard LU codes)
        dor_lu_tbl (str; Path): path to dor use code table (maps DOR LU codes to more standard and generalized categories)
        mobility_fee_df (pd.Dataframe): mobility fee dataframe
        out_file (str, Path): path to output permit csv
        out_crs (int): EPSG code of output dataset

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
    permit_df = permit_df.merge(
        right=mobility_fee_df, how="inner", left_on="CAT_CODE", right_on="Code"
    )

    # convert to points
    sdf = add_xy_from_poly(
        poly_fc=parcel_fc,
        poly_key=poly_key,
        table_df=permit_df,
        table_key=permit_key,
        to_crs=out_crs,
    )
    sdf.fillna(0.0, inplace=True)

    sdf = gpd.GeoDataFrame(sdf, geometry=sdf["geometry"], crs=out_crs)
    conversion = {col: float for col in sdf.columns if is_numeric_dtype(sdf[col])}
    sdf = sdf.astype(conversion)
    sdf.to_file(out_file)
    return sdf


def lu_cost_alloc_factor(val_list):
    total = abs(sum(val_list))
    return [val / total for val in val_list]


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
    mobility_fee_data = Path(PROJ_PATH, "docs", "MobilityFeeCalculation_v5a.xlsx")
    sheet_name = "FEE_ByContextArea_ByMode_CODE"
    column_names = ["Code"] + \
                   [f"{col}_{mode}" for col in ["R1_In", "R2_In", "R3_In", "R4_In", "R2_Out", "R3_Out", "R4_Out", ]
                    for mode in ["Road", "Transit", "Walk_Bike"]]

    mf_sched_df = pd.read_excel(
        io=mobility_fee_data, sheet_name=sheet_name, header=0, usecols=column_names
    )
    # mf_sched_df = mf_sched_df[column_names]
    rif_sched_df = pd.read_csv(rif_schedule_tbl)

    out_folder = Path(PROJ_PATH, "outputs", "version13_splitMode")
    if not out_folder.exists():
        out_folder.mkdir()
    dfs = []
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:  # , 2016, 2017, 2018, 2019, 2020
        pdc_mult = fee_scaler[year]
        permits = list(raw_permits.glob(f"*{year}.csv"))
        out_file = f"{permits[0].stem}_cleaned"
        parcels = make_path(parcel_gdbs, f"PMT_{year}.gdb", "Polygons", "Parcels")
        if year == 2020:
            parcels = make_path(parcel_gdbs, f"PMT_2019.gdb", "Polygons", "Parcels")

        # scale rif schedule values by year
        rif_df = rif_sched_df.copy()
        rif_df["fee_outside"] *= pdc_mult
        rif_df["fee_inside"] *= pdc_mult

        # scale mf schedule values by year
        mf_df = mf_sched_df.copy()
        # mf_df[
        #     ["R1_In", "R2_In", "R2_Out", "R3_In", "R3_Out", "R4_In", "R4_Out"]
        # ]  *= pdc_mult

        # tidy permit data and geo-reference it to parcel centroids
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
        permit_gdf = permit_gdf.merge(
            right=rif_df, left_on="CAT_CODE", right_on="CODE", how="inner"
        )

        # id points inside and outside urban infill area
        permit_gdf["infill"] = permit_gdf.intersects(infill_gdf.unary_union)

        # intersect permits with context areas and add RIF/MF comparison columns to get summary data
        intersect = pd.DataFrame(
            gpd.overlay(df1=summary_gdf, df2=permit_gdf, keep_geom_type=False)
        )
        # drop rows where Admin fee is greater than 5% of Construction Fee
        intersect = intersect[intersect["ADMIN_COST"] < (intersect["CONST_COST"] * 0.05)]

        # calculate RIF from schedules and apply ped discount
        intersect["RIF_CALCULATED"] = np.where(
            intersect["infill"] == True,
            intersect["fee_inside"] * intersect["UNITS_VAL"],
            intersect["fee_outside"] * intersect["UNITS_VAL"],
        )
        intersect["RIF_CALCULATED"].loc[intersect["PED_ORIENTED"] == 1] = (
                intersect["RIF_CALCULATED"] * 0.859
        )  # reduction credit applied for pedestrian oriented projects

        # calculate MF from schedules and apply ped discount
        modes = ["Road", "Transit", "Walk_Bike"]
        mf_calc_modes = [f"MF_CALC_{mode}" for mode in modes]
        for mode in mf_calc_modes:
            intersect[mode] = 0.0
        for ix, row in summary_gdf.iterrows():
            ring = row["Ring"]
            smart = row["SMART"]
            for mode in modes:
                intersect[f"MF_CALC_{mode}"].loc[
                    (intersect["Ring"] == ring) & (intersect["SMART"] == smart)
                    ] = (intersect[f"R{ring}_{smart}_{mode}"] * intersect["UNITS_VAL"])
        intersect["MF_CALCULATED"] = intersect[mf_calc_modes].sum(axis=1)

        # reduction credit applied for pedestrian oriented projects
        for mode in mf_calc_modes + ["MF_CALCULATED"]:
            intersect[mode].loc[intersect["PED_ORIENTED"] == 1] = (
                    intersect[mode] * 0.859
            )
        # difference of
        intersect[f"DIFF_OF_CALCULATED"] = (
                    intersect["RIF_CALCULATED"] - intersect[f"MF_CALCULATED"]
            )

        # back into proportion attributed to each LU using a lu_cost_allocation calc
        gb_sums = intersect.groupby("PROC_NUM").agg({"RIF_CALCULATED": lambda s: abs(s.sum())})
        gb_sums.rename(columns={"RIF_CALCULATED": "rif_sums"}, inplace=True)
        intersect = intersect.merge(gb_sums, left_on="PROC_NUM", right_index=True)
        intersect["rif_factor"] = (intersect["CONST_COST"] + intersect["ADMIN_COST"] + intersect["CREDITS"]
                                   ) / intersect["rif_sums"]
        intersect["RIF_ACTUAL"] = intersect['RIF_CALCULATED'] * intersect["rif_factor"]
        # intersect["RIF_ACTUAL_ORIGINAL"] = (intersect["CONST_COST"] + intersect["ADMIN_COST"] + intersect["CREDITS"])

        # summarize data by Context Area and various Landuse types
        groupby_cols = ["Ring", "SMART", "STATUS", "PED_ORIENTED", "PROC_NUM",
                        "CAT_CODE", "LANDUSE", "SPC_LU", "GN_VA_LU", "UNITS_VAL", "UNITS",]
        sum_cols = ["RIF_CALCULATED"] + mf_calc_modes + ["MF_CALCULATED", "RIF_ACTUAL"]
        df = (intersect.groupby(groupby_cols)[sum_cols].sum().reset_index())

        df["DIFF_RFc_RFa"] = df["RIF_CALCULATED"] - df["RIF_ACTUAL"]
        df["DIFF_MFc_RFc"] = df["MF_CALCULATED"] - df["RIF_CALCULATED"]
        df["DIFF_MFc_RFa"] = df["MF_CALCULATED"] - df["RIF_ACTUAL"]
        df["YEAR"] = year
        df["SPC_LU"].replace({"0": "Unknown"}, inplace=True)
        df["GN_VA_LU"].replace({"0": "Unknown"}, inplace=True)
        df.to_csv(path_or_buf=os.path.join(out_folder, f"{out_file}.csv"))
        dfs.append(df)

    # write out summary data
    out_data = pd.concat(dfs)
    # out_data.to_csv(os.path.join(out_folder, "RIF_summary_2015-2020.csv"))
    date_str = datetime.datetime.now().strftime("%m%d%y")
    out_data.to_excel(
        Path(out_folder, f"RIF_summary_2015-2020_{date_str}.xlsx"), sheet_name="Summary_2015-2020"
    )
