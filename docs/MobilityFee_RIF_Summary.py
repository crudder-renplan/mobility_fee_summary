# %% IMPORTS
import geopandas as gpd
import fiona
import pandas as pd
import os
from pathlib import Path

# %% READ DATA
gdb_path = r"C:\OneDrive_RP\OneDrive - Renaissance Planning Group\SHARE\Mobility_Fee(1)\April21_Buffer.gdb"
layers = fiona.listlayers(gdb_path)
permit_path = Path(r"C:\Users\V_RPG\Desktop\Miami-Dade MobilityFee - Permits")

permits = permit_path.glob( "*.shp")
summary_gdf = gpd.read_file(gdb_path, driver="FileGDB", layer='Context_Area_Union')
summary_gdf = summary_gdf.replace({"SMART": {0: "Out", 1: "In"}})

dfs=[]
for permit in permits:
    out_file = permit.stem
    year = out_file[-4:]
    print(out_file)
    permit_gdf = gpd.read_file(permit)
    permit_gdf.to_crs(crs=summary_gdf.crs, inplace=True)

    # zero out duplicated costs to 0 so the values arent repeated
    permit_gdf.loc[permit_gdf.assign(d=permit_gdf.COST).duplicated(["PROC_NUM", "FOLIO"]), "COST"] = 0.0

    # %% INTERSECT
    if year == "2020":
        print()
    intersect = pd.DataFrame(gpd.overlay(df1=summary_gdf, df2=permit_gdf, keep_geom_type=False))
    df = intersect.groupby(["Ring", "SMART", "PED_ORIENT", "LANDUSE", "SPC_LU", "GN_VA_LU", "UNITS"])["UNITS_VAL", "COST"].sum().reset_index()
    df["YEAR"] = year
    # df.rename(columns={"SPC_LU":"LANDUSE"}, inplace=True)
    df["SPC_LU"].replace({"0": "Unknown"}, inplace=True)
    df["GN_VA_LU"].replace({"0": "Unknown"}, inplace=True)
    df.to_csv(path_or_buf=os.path.join(permit_path, f"{out_file}.csv"))
    dfs.append(df)

out_data = pd.concat(dfs)
out_data.to_csv(os.path.join(permit_path, "RIF_summary_2015-2020.csv"))