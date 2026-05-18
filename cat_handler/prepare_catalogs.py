
import cat_handler
from cat_handler import paths
from cat_handler.parsers import anss, gcmt, isc, isc_gem

def main():

    # anss.prepare_anss(paths.rawcat_anss, "cat_valp.csv")
    gcmt.prepare_gcmt(paths.rawcat_gcmt, paths.rawcat_gcmt_iris, paths.cat_gcmt)


if __name__ == '__main__':
    main()