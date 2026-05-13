"""
project: 2d-cnn image chip generation
topic: geometric tiling of irregular segments for deep learning
author: lasya venigalla, ph.d.
"""

import os
import pycrs
import fiona
import rasterio
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr
from shapely import geometry
from shapely.geometry import Point, Polygon, mapping
from rasterio.mask import mask

# --- 1. geometric utilities ---

class bbox_segment(object):
    """calculates the minimum bounding box for a set of points."""
    def __init__(self, points):
        if not points:
            raise ValueError("list of points is empty")
        self.minx = min(p[0] for p in points)
        self.miny = min(p[1] for p in points)
        self.maxx = max(p[0] for p in points)
        self.maxy = max(p[1] for p in points)

def get_seg_geometry(lyr, feat_idx):
    """retrieves polygon ring points from an ogr layer."""
    layer = lyr.GetLayer()
    feature = layer.GetFeature(feat_idx)
    geom = feature.GetGeometryRef().GetGeometryRef(0)
    return [(geom.GetX(p), geom.GetY(p)) for p in range(geom.GetPointCount())]

def calculate_grid(lyr, feat_idx, step_size, res, num_pixels):
    """generates equidistant grid points centered on a segment's centroid."""
    points = get_seg_geometry(lyr, feat_idx)
    bbox = bbox_segment(points)
    
    # calculate centroid
    cent_x = sum(p[0] for p in points) / len(points)
    cent_y = sum(p[1] for p in points) / len(points)

    # generate grid ranges
    k = step_size / res
    x_range = np.unique(np.concatenate((np.arange(cent_x, bbox.minx - k, -step_size), 
                                        np.arange(cent_x, bbox.maxx + k, step_size))))
    y_range = np.unique(np.concatenate((np.arange(cent_y, bbox.miny - k, -step_size), 
                                        np.arange(cent_y, bbox.maxy + k, step_size))))
    
    grid_pts = [[x, y] for x in x_range for y in y_range]
    
    # create square polygon vertices for each grid point
    sq_len = res * num_pixels
    poly_pts_list = []
    for pt in grid_pts:
        half = sq_len / 2
        # coordinates for square corners
        p1 = [pt[0] - half, pt[1] - half]
        p2 = [pt[0] + half, pt[1] - half]
        p3 = [pt[0] - half, pt[1] + half]
        p4 = [pt[0] + half, pt[1] + half]
        poly_pts_list.append([p4, p3, p1, p2])
        
    return grid_pts, poly_pts_list

# --- 2. data export functions ---

def save_geometries(points, polygons, seg_id, class_name, path_pts, path_polys, epsg=32138):
    """saves generated grid points and polygons to shapefiles."""
    schema_pts = {'geometry': 'Point', 'properties': {'id': 'int', 'seg_id': 'int', 'class': 'str'}}
    schema_poly = {'geometry': 'Polygon', 'properties': {'id': 'int', 'seg_id': 'int', 'class': 'str'}}
    
    pts_file = f"{path_pts}pts_{seg_id}_{class_name}.shp"
    poly_file = f"{path_polys}poly_{seg_id}_{class_name}.shp"

    with fiona.open(pts_file, 'w', 'ESRI Shapefile', schema_pts, crs=f'epsg:{epsg}') as c:
        for i, pt in enumerate(points):
            c.write({'geometry': mapping(Point(pt[0], pt[1])), 
                     'properties': {'id': i, 'seg_id': seg_id, 'class': class_name}})

    with fiona.open(poly_file, 'w', 'ESRI Shapefile', schema_poly, crs=f'epsg:{epsg}') as c:
        for i, poly in enumerate(polygons):
            # convert list of points to shapely polygon
            sh_poly = Polygon(poly)
            c.write({'geometry': mapping(sh_poly), 
                     'properties': {'id': i, 'seg_id': seg_id, 'class': class_name}})

def export_image_chips(poly_shp_path, src_raster, output_base_path, is_train, is_test):
    """crops the master raster into small chips based on grid polygons."""
    gdf = gpd.read_file(poly_shp_path)
    seg_id = list(set(gdf.get('seg_id')))[0]
    class_name = list(set(gdf.get('class')))[0]

    # determine destination folder
    sub_folder = "Training" if is_train else ("Testing" if is_test else "Classification")
    final_dir = os.path.join(output_base_path, sub_folder, class_name)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    for i, geom in enumerate(gdf.geometry.values):
        out_img, out_trans = mask(src_raster, [mapping(geom)], crop=True, filled=False)
        out_meta = src_raster.meta.copy()
        
        # enforce 16x16 dimension
        h, w = 16, 16
        epsg = int(src_raster.crs.data['init'][5:])
        
        out_meta.update({
            "driver": "GTiff", "height": h, "width": w,
            "transform": out_trans, "crs": pycrs.parse.from_epsg_code(epsg).to_proj4()
        })

        out_path = f"{final_dir}/img_{seg_id}_{class_name}_{i}.tif"
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_img)

# --- 3. main execution logic ---

def run_chip_generation(in_seg_path, path_pts, path_polys, raster_src, res, pixels, is_train=True, is_test=False):
    """orchestrates the full geometry-to-chip workflow."""
    layer = in_seg_path.GetLayer()
    step_size = res * pixels
    
    for i in range(layer.GetFeatureCount()):
        feat = layer.GetFeature(i)
        seg_id = feat.GetField("Id")
        class_name = feat.GetField("Class_name") if (is_train or is_test) else "cls"
        
        # 1. generate grids
        grid_pts, poly_list = calculate_grid(in_seg_path, i, step_size, res, pixels)
        
        # 2. save vector grids
        save_geometries(grid_pts, poly_list, seg_id, class_name, path_pts, path_polys)
        
        # 3. generate raster chips
        poly_path = f"{path_polys}poly_{seg_id}_{class_name}.shp"
        export_image_chips(poly_path, raster_src, "C:\\PhDProjects\\Output\\", is_train, is_test)

# --- initialization ---
if __name__ == "__main__":
    img_path = r'C:\PhDProjects\WVDallasComposite.tif'
    with rasterio.open(img_path) as src:
        resolution = src.transform[0]
        num_pixels = 16
        
        # example: run for training segments
        # train_seg_path = ogr.Open(r'C:\PhDProjects\2dcnn\TrainingSegments.shp')
        # run_chip_generation(train_seg_path, "path/to/pts/", "path/to/poly/", src, resolution, num_pixels)
