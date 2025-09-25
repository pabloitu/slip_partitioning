
import cat_handler
from cat_handler import paths
from cat_handler.parsers import anss, gcmt, isc, isc_gem

def main():

    # isc_priority = {"GFZ": 0, "GEOFON": 0, "NEIC": 1, "NEIS": 1, "US": 1, "USGS": 1}
    # isc.prepare_isc(paths.rawcat_isc, paths.cat_isc, isc_priority)
    isc_gem.prepare_isc_gem(paths.rawcat_isc_gem, paths.cat_isc_gem)
    # anss.prepare_anss(paths.rawcat_anss, paths.cat_anss)
    # gcmt.prepare_gcmt(paths.rawcat_gcmt, paths.rawcat_gcmt_iris, paths.cat_gcmt)
    # gcmt.prepare_gmt_nico(paths.rawcat_gcmt_perez, paths.cat_gcmt_perez)


if __name__ == '__main__':
    main()