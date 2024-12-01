# Dataset Overview

This dataset, used for generating patches, consists of various geospatial and geomorphological attributes derived from raster files. The patches are intended for training and evaluating machine learning models on hydrological streamline delineation and geologic map segmentation tasks. 

## Data Structure and Channels

Each patch is constructed using a stack of TIFF files, each representing a different feature layer. Depending on the dataset directory (`AK_50_Dataset` or others), the dataset includes different versions of orientation data (`ori` vs `ori_ave`). The final channel in each stack is a mask, which is used as the label for training.

### Channels and Files

The TIFF files are named according to the HUC (Hydrologic Unit Code) identifier for each watershed. Below is a description of each channel and the corresponding file used:

1. **Curvature**: Curvature of the terrain, providing information about surface shape.
   - Filename: `curvature_{huc_code}.tif`

2. **SWM1**: Stream power index metric 1, a measure related to water flow energy.
   - Filename: `swm1_{huc_code}.tif`

3. **SWM2**: Stream power index metric 2, another measure related to water flow.
   - Filename: `swm2_{huc_code}.tif`

4. **Orientation**: Terrain orientation data. This differs between datasets:
   - For `AK_50_Dataset`: `ori_{huc_code}.tif`
   - For other datasets: `ori_ave_{huc_code}.tif`

5. **DSM**: Digital Surface Model, representing the earth’s surface and vegetation/building heights.
   - Filename: `dsm_{huc_code}.tif`

6. **Geomorphology**: Information on the geomorphological aspects of the terrain.
   - Filename: `geomorph_{huc_code}.tif`

7. **Positive Openness**: Measure of terrain exposure.
   - Filename: `pos_openness_{huc_code}.tif`

8. **TPI 11**: Topographic Position Index with a neighborhood size of 11, indicating ridges and valleys.
   - Filename: `tpi_11_{huc_code}.tif`

9. **TWI**: Topographic Wetness Index, indicating soil moisture content.
   - Filename: `twi_{huc_code}.tif`

10. **TPI 3**: Topographic Position Index with a neighborhood size of 3.
    - Filename: `tpi_3_{huc_code}.tif`

11. **DTM**: Digital Terrain Model, representing ground elevations.
    - Filename: `dtm_{huc_code}.tif`

12. **Mask**: Binary mask file indicating areas of interest. This serves as the label for model training.
    - Extracted directly from the TIFF mask file for each HUC code.

### Reference Data Preprocessing

The reference mask data is processed according to the following function:

```python
def preprocess_label_tif(tif_data):
    tif_data[tif_data != 0] = 1
    return tif_data