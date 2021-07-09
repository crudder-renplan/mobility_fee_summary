import os
import uuid
import arcpy
import pandas as pd
import numpy as np
from arcgis import GeoAccessor, GeoSeriesAccessor
from pathlib import Path

import geopandas as gpd
import fiona
import pandas as pd

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


def make_inmem_path(file_name=None):
    """Generates an in_memory path usable by arcpy that is unique to avoid any overlapping names. If a file_name is
    provided, the in_memory file will be given that name with an underscore appended to the beginning.

    Returns:
        String; in_memory path

    Raises:
        ValueError, if file_name has been used already
    """
    if not file_name:
        unique_name = f"_{str(uuid.uuid4().hex)}"
    else:
        unique_name = f"_{file_name}"
    try:
        in_mem_path = make_path("in_memory", unique_name)
        if arcpy.Exists(in_mem_path):
            raise ValueError
        else:
            return in_mem_path
    except ValueError:
        print("The file_name supplied already exists in the in_memory space")


def check_overwrite_output(output, overwrite=False):
    """A helper function that checks if an output file exists and
        deletes the file if an overwrite is expected.

    Args:
        output: Path
            The file to be checked/deleted
        overwrite: Boolean
            If True, `output` will be deleted if it already exists.
            If False, raises `RuntimeError`.

    Raises:
        RuntimeError:
            If `output` exists and `overwrite` is False.
    """
    if arcpy.Exists(output):
        if overwrite:
            print(f"--- --- deleting existing file {output}")
            arcpy.Delete_management(output)
        else:
            raise RuntimeError(f"Output file {output} already exists")


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


def polygons_to_points(in_fc, out_fc, fields="*", skip_nulls=False, null_value=0):
    """Convenience function to dump polygon features to centroids and save as a new feature class.

    Args:
         in_fc (str): Path to input feature class
         out_fc (str): Path to output point fc
         fields (str or list, default="*"): [String,...] fields to include in conversion
         skip_nulls (bool, default=False): Control whether records using nulls are skipped.
         null_value (int, default=0): Replaces null values from the input with a new value.

     Returns:
         out_fc (str): path to output point feature class
    """
    # TODO: adapt to search-cursor-based derivation of polygon.centroid to ensure point is within polygon
    sr = arcpy.Describe(in_fc).spatialReference
    if fields == "*":
        fields = [f.name for f in arcpy.ListFields(in_fc) if not f.required]
    elif isinstance(fields, str):
        fields = [fields]
    fields.append("SHAPE@XY")
    a = arcpy.da.FeatureClassToNumPyArray(
        in_table=in_fc, field_names=fields, skip_nulls=skip_nulls, null_value=null_value
    )
    arcpy.da.NumPyArrayToFeatureClass(
        in_array=a, out_table=out_fc, shape_fields="SHAPE@XY", spatial_reference=sr
    )
    fields.remove("SHAPE@XY")
    return out_fc


def add_xy_from_poly(poly_fc, poly_key, table_df, table_key):
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
    pts = polygons_to_points(
        in_fc=poly_fc, out_fc=make_inmem_path(), fields=poly_key, null_value=0.0
    )
    pts_sdf = pd.DataFrame.spatial.from_featureclass(pts)

    esri_ids = ["OBJECTID", "FID"]
    if any(item in pts_sdf.columns.to_list() for item in esri_ids):
        pts_sdf.drop(labels=esri_ids, axis=1, inplace=True, errors="ignore")
    # join permits to parcel points MANY-TO-ONE
    print("--- merging geo data to tabular")
    # pts = table_df.merge(right=pts_sdf, how="inner", on=table_key)
    return pd.merge(
        left=table_df, right=pts_sdf, how="inner", left_on=table_key, right_on=poly_key
    )


def df_to_points(df, out_fc, shape_fields, from_sr, to_sr, overwrite=False):
    """Use a pandas data frame to export an arcgis point feature class.

    Args:
        df (pandas.DataFrame): A dataframe with valid `shape_fields` that can be
            interpreted as x,y coordinates
        out_fc (str): Path to the point feature class to be generated.
        shape_fields (list): Columns to be used as shape fields (x, y)
        from_sr (arcpy.SpatialReference): The spatial reference definition for
            the coordinates listed in `shape_field`
        to_sr (arcpy.SpatialReference): The spatial reference definition
            for the output features.
        overwrite (bool, default=False)

    Returns:
        out_fc (str): Path
    """
    # set paths
    temp_fc = make_inmem_path()

    # coerce sr to Spatial Reference object
    # Check if it is a spatial reference already
    try:
        # sr objects have .type attr with one of two values
        check_type = from_sr.type
        type_i = ["Projected", "Geographic"].index(check_type)
    except:
        from_sr = arcpy.SpatialReference(from_sr)
    try:
        # sr objects have .type attr with one of two values
        check_type = to_sr.type
        type_i = ["Projected", "Geographic"].index(check_type)
    except:
        to_sr = arcpy.SpatialReference(to_sr)

    # build array from dataframe
    in_array = np.array(np.rec.fromrecords(df.values, names=df.dtypes.index.tolist()))
    # write to temp feature class
    arcpy.da.NumPyArrayToFeatureClass(
        in_array=in_array,
        out_table=temp_fc,
        shape_fields=shape_fields,
        spatial_reference=from_sr,
    )
    # reproject if needed, otherwise dump to output location
    if from_sr != to_sr:
        arcpy.Project_management(
            in_dataset=temp_fc, out_dataset=out_fc, out_coor_system=to_sr
        )
    else:
        out_path, out_fc = os.path.split(out_fc)
        if overwrite:
            check_overwrite_output(output=out_fc, overwrite=overwrite)
        arcpy.FeatureClassToFeatureClass_conversion(
            in_features=temp_fc, out_path=out_path, out_name=out_fc
        )
    # clean up temp_fc
    arcpy.Delete_management(in_data=temp_fc)
    return out_fc


# permit functions
def clean_permit_data(
    permit_csv,
    parcel_fc,
    permit_key,
    poly_key,
    rif_lu_tbl,
    dor_lu_tbl,
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
    permit_df[permit_key] = permit_df[permit_key].astype(np.str)
    permit_df[poly_key] = permit_df[permit_key].apply(lambda x: x.zfill(13))
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
    permit_df["COST"] = permit_df["CONST_COST"] + permit_df["ADMIN_COST"]
    #   id project as pedestrain oriented
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
    lu_df = lu_df.merge(right=dor_df, how="inner", on="DOR_UC")
    permit_df = permit_df.merge(right=lu_df, how="inner", on="CAT_CODE")
    #   drop unnecessary columns
    permit_df.drop(columns=PERMITS_DROPS, inplace=True)

    # convert to points
    sdf = add_xy_from_poly(
        poly_fc=parcel_fc, poly_key=poly_key, table_df=permit_df, table_key=permit_key
    )
    sdf.fillna(0.0, inplace=True)

    # get xy coords and drop SHAPE
    sdf["X"] = sdf[sdf.spatial.name].apply(lambda c: c.x)
    sdf["Y"] = sdf[sdf.spatial.name].apply(lambda c: c.y)

    # TODO: assess why sanitize columns as false fails (low priority)
    # sdf.spatial.to_featureclass(location=out_file, sanitize_columns=False)
    df_to_points(
        df=sdf.drop(columns="SHAPE", inplace=True),
        out_fc=out_file,
        shape_fields=["X", "Y"],
        from_sr=out_crs,
        to_sr=out_crs,
    )
    return sdf


if __name__ == "__main__":
    # raw data files/paths
    data_folder = Path(r"C:\Users\V_RPG\Desktop\Miami-Dade MobilityFee - Permits")
    raw_permits = Path(data_folder, "source_reports")
    RIF_CAT_CODE_TBL = Path(data_folder, "docs", "road_impact_fee_cat_codes.csv")
    DOR_UC_TBL = Path(data_folder, "docs", "Land_Use_Recode.csv")

    # summary data files/paths
    gdb_path = r"C:\OneDrive_RP\OneDrive - Renaissance Planning Group\SHARE\Mobility_Fee(1)\April21_Buffer.gdb"
    layers = fiona.listlayers(gdb_path)
    permit_path = Path(r"C:\Users\V_RPG\Desktop\Miami-Dade MobilityFee - Permits")
    summary_gdf = gpd.read_file(gdb_path, driver="FileGDB", layer="Context_Area_Union")
    summary_gdf = summary_gdf.replace({"SMART": {0: "Out", 1: "In"}})

    # parcel data files/paths
    parcel_gdbs = r"C:\OneDrive_RP\OneDrive - Renaissance Planning Group\SHARE\PMT_link\Data\CLEANED"

    # ratio data files/paths
    ratio_file = make_path(data_folder, "MobilityFeeCalculation_v3_chs.xlsb.xlsx")
    sheet_name = "RIF_FEE_RATIO"
    column_names = [
        "ITE Land Use Code",
        "Land Use",
        "Unit",
        "Printed RIF (Current Fee)",
        # "R1_Within SMART Corridor_FEE",  "R1_Outside SMART Corridor_FEE",
        # "R2_Within Smart Corridor_FEE", "R2_Outside Smart Corridor_FEE",
        # "R3_Within Smart Corridor_FEE", "R3_Outside Smart Corridor_FEE",
        # "R4_Within Smart Corridor_FEE", "R4_Outside Smart Corridor_FEE",
        "R1_Within SMART Corridor_RATIO",  # "R1_Outside SMART Corridor_RATIO",
        "R2_Within Smart Corridor_RATIO",
        "R2_Outside Smart Corridor_RATIO",
        "R3_Within Smart Corridor_RATIO",
        "R3_Outside Smart Corridor_RATIO",
        "R4_Within Smart Corridor_RATIO",
        "R4_Outside Smart Corridor_RATIO",
    ]
    ratio_df = pd.read_excel(
        filepath=ratio_file, sheet_name=sheet_name, header=0, usecols=column_names
    )

    out_folder = Path(data_folder, "outputs", "version2")
    dfs = []
    for year in range(2015, 2019):
        permits = data_folder.glob(f"*{year}.csv")
        out_file = permits[0].stem
        parcels = make_path(parcel_gdbs, f"PMT_{year}.gdb", "Polygons", "Parcels")
        permit_sdf = clean_permit_data(
            permit_csv=permits[0],
            parcel_fc=parcels,
            permit_key="FOLIO_NUM",
            poly_key="FOLIO",
            rif_lu_tbl=RIF_CAT_CODE_TBL,
            dor_lu_tbl=DOR_UC_TBL,
            out_file=make_path(out_folder, "RIF_Permits_{}.shp".format(year)),
            out_crs=2881,
        )
        permit_gdf = gpd.GeoDataFrame(permit_sdf, crs=2881, geometry="SHAPE")
        permit_gdf.to_crs(crs=summary_gdf.crs, inplace=True)
        permit_gdf = permit_gdf.join(ratio_df, on="CAT_CODE", how="inner")
        # zero out duplicated costs to 0 so the values arent repeated
        permit_gdf.loc[
            permit_gdf.assign(d=permit_gdf.COST).duplicated(["PROC_NUM", "FOLIO"]),
            "COST",
        ] = 0.0

        # intersect permits with context areas and add RIF/MF comparison columns to get summary data
        intersect = pd.DataFrame(
            gpd.overlay(df1=summary_gdf, df2=permit_gdf, keep_geom_type=False)
        )
        for ix, row in summary_gdf.iterrows():
            ring = row["Ring"]
            smart = row["SMART"]
            if smart == 1:
                smart = "Within"
            if smart == 0:
                smart = "Outside"
            intersect["RATIO"] = np.where(
                intersect["Ring"] == ring & intersect["SMART"] == smart,
                intersect[f"R{ring}_{smart}Corridor_RATIO"],
            )
        intersect["MOBILITY_FEE"] = intersect["COST"] * intersect["RATIO"]
        intersect["DIFF_MF_vs_RF"] = intersect["MOBILITY_FEE"] - intersect["COST"]

        # summarize data by Context Area and various Landuse types
        df = (
            intersect.groupby(
                [
                    "Ring",
                    "SMART",
                    "PED_ORIENT",
                    "CAT_CODE",
                    "LANDUSE",
                    "SPC_LU",
                    "GN_VA_LU",
                    "UNITS",
                ]
            )["UNITS_VAL", "COST", "RATIO", "MOBILITY_FEE", "DIFF_MF_vs_RF"]
            .sum()
            .reset_index()
        )
        df["YEAR"] = year
        # df.rename(columns={"SPC_LU":"LANDUSE"}, inplace=True)
        df["SPC_LU"].replace({"0": "Unknown"}, inplace=True)
        df["GN_VA_LU"].replace({"0": "Unknown"}, inplace=True)
        df.to_csv(path_or_buf=os.path.join(out_folder, f"{out_file}.csv"))
        dfs.append(df)

    # write out summary data
    out_data = pd.concat(dfs)
    out_data.to_csv(os.path.join(out_folder, "RIF_summary_2015-2020.csv"))
