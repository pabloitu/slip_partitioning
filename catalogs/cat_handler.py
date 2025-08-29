import pandas
import csep
from csep.core import regions
import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


initial_polygon = gpd.read_file('../polygons/intraarc_polygon.shp')
initial_polygon = initial_polygon.union_all()

#### Parse GMT
def parse_gcmt(plot=True):
    data_gmt = pandas.read_csv('./global/GMT_1976_2025_consolidado_conSlab.csv')

    times = []
    data = []
    for i, j in data_gmt.iterrows():
        dt = datetime.timedelta(days=i)
        event = (f'{i}', csep.utils.time_utils.datetime_to_utc_epoch(datetime.datetime(1970, 1, 1) + dt), j.Lat, j.Lon, j.Depth_km, j.Mw)
        data.append(event)

    # filter by polygon
    points = gpd.GeoSeries([Point(i[3], i[2]) for i in data])
    inside_flags = points.within(initial_polygon)
    data = [i for i, j in zip(data, inside_flags) if j]

    catalog = csep.core.catalogs.CSEPCatalog(data=data)
    catalog.filter(['magnitude >= 5.0', 'depth <= 25'])

    return catalog

#### Parse USGS
def parse_usgs(plot=True):
    data_gmt = pandas.read_csv('./global_2/anss.csv')

    times = []
    data = []
    for i, j in data_gmt.iterrows():
        event = (f'{i}', csep.utils.time_utils.datetime_to_utc_epoch(datetime.datetime.fromisoformat(j.time)), j.latitude, j.longitude, j.depth, j.mag)
        data.append(event)

    # filter by polygon
    points = gpd.GeoSeries([Point(i[3], i[2]) for i in data])
    inside_flags = points.within(initial_polygon)
    data = [i for i, j in zip(data, inside_flags) if j]

    catalog = csep.core.catalogs.CSEPCatalog(data=data)
    catalog.filter(['magnitude >= 5.0', 'depth <= 25'])


    return catalog

def parse_isc(plot=True):
    data_gmt = pandas.read_csv('./global/isc-gem-cat.csv', delimiter=',')
    data_gmt.columns = data_gmt.columns.str.strip()

    # 2) Strip spaces from string/object columns
    data_gmt = data_gmt.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    times = []
    data = []
    for i, j in data_gmt.iterrows():
        event = (f'{i}', csep.utils.time_utils.datetime_to_utc_epoch(datetime.datetime.fromisoformat(j.date)), j.lat, j.lon, j.depth, j.mw)
        data.append(event)

    # filter by polygon
    points = gpd.GeoSeries([Point(i[3], i[2]) for i in data])
    inside_flags = points.within(initial_polygon)
    data = [i for i, j in zip(data, inside_flags) if j]

    catalog = csep.core.catalogs.CSEPCatalog(data=data)
    catalog.filter(['magnitude >= 5.0', 'depth <= 25'])


    return catalog

if __name__ == '__main__':
    cat_gcmt = parse_gcmt()
    cat_usgs = parse_usgs()
    cat_isc = parse_isc()



    ax = cat_usgs.plot(extent=[-77, -67.5, -50, -28], plot_args={'markercolor': 'red'})
    ax = cat_gcmt.plot(ax=ax,plot_args={'markercolor':'blue'})
    cat_isc.plot(ax=ax, plot_args={'markercolor': 'green'}, show=True)
