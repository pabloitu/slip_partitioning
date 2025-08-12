import pandas
import csep
import datetime
import matplotlib.pyplot as plt
data_gmt = pandas.read_csv('./global/GMT_1976_2025_consolidado_conSlab.csv')

times = []
data = []
for i, j in data_gmt.iterrows():
    dt = datetime.timedelta(days=i)
    # times.append(da)
    event = (f'{i}', csep.utils.time_utils.datetime_to_utc_epoch(datetime.datetime(1970, 1, 1) + dt), j.Lat, j.Lon, j.Depth_km, j.Mw)
    data.append(event)
    # print(i,j )
catalog = csep.core.catalogs.CSEPCatalog(data=data)
catalog.filter_spatial
catalog.plot(show=True)