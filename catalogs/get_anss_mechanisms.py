# build_anss_mechanisms.py
# Requirements: requests, numpy
# Output: an NDK-like mechanisms CSV matching your GCMT file fields.
import pandas
from obspy.clients.fdsn import Client
import numpy as np

import csv
import math
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple, List
#

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def axes_from_sdr(strike: float, dip: float, rake: float):
    """
    (strike, dip, rake) -> dict with T/N/P axes (azimuth, plunge) in GCMT convention.
    Azimuth: CW from North; Plunge: downward 0..90.
    """
    st, di, ra = map(np.radians, [strike, dip, rake])
    # plane normal n and slip s in NED
    nN = -np.sin(di) * np.sin(st)
    nE =  np.sin(di) * np.cos(st)
    nD = -np.cos(di)
    sN =  np.cos(ra) * np.cos(st) + np.sin(ra) * np.cos(di) * np.sin(st)
    sE =  np.cos(ra) * np.sin(st) - np.sin(ra) * np.cos(di) * np.cos(st)
    sD =  np.sin(ra) * np.sin(di)
    n = np.array([nN, nE, nD]); n /= np.linalg.norm(n)
    s = np.array([sN, sE, sD]); s /= np.linalg.norm(s)
    M = np.outer(s, n) + np.outer(n, s)  # DC tensor (orientation only)
    # eigenvectors: columns correspond to P (min), N (mid), T (max)
    w, V = np.linalg.eigh(M)
    idx = np.argsort(w)
    V = V[:, idx]
    P, N, T = V[:, 0], V[:, 1], V[:, 2]

    def vec_to_az_pl(v):
        v = v / np.linalg.norm(v)
        if v[2] < 0:  # force downward plunge
            v = -v
        az = (math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0
        pl = math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))
        return az, pl

    T_az, T_pl = vec_to_az_pl(T)
    N_az, N_pl = vec_to_az_pl(N)
    P_az, P_pl = vec_to_az_pl(P)
    return {
        "T_plunge": T_pl, "T_azimuth": T_az,
        "N_plunge": N_pl, "N_azimuth": N_az,
        "P_plunge": P_pl, "P_azimuth": P_az,
    }

def pick_product(prod_list, prefer_source="us"):
    """
    Choose a product dict from a list of GeoJSON products.
    Priority: source==prefer_source -> preferred==True -> first item.
    """
    if not prod_list:
        return None
    if prefer_source:
        for p in prod_list:
            if p.get("source") == prefer_source:
                return p
    for p in prod_list:
        if p.get("preferred"):
            return p
    return prod_list[0]

def fetch_products_from_comcat(event_id: str):
    """
    GeoJSON detail call for one event id; returns the 'products' dict.
    """
    params = {"format": "geojson", "eventid": event_id, "includesuperseded": "true"}
    r = requests.get(USGS_BASE, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    feats = js.get("features") or []
    if not feats:
        return {}
    return (feats[0].get("properties") or {}).get("products") or {}

def extract_mech_from_products(products: dict):
    """
    Pull planes (from 'focal-mechanism' or 'moment-tensor') and axes (from 'moment-tensor').
    Returns dict with strike1,dip1,rake1, strike2,dip2,rake2, and optionally T/N/P.
    """
    fm = pick_product(products.get("focal-mechanism", []), prefer_source="us")
    mt = pick_product(products.get("moment-tensor", []),   prefer_source="us")
    fm_p = fm.get("properties") if fm else {}
    mt_p = mt.get("properties") if mt else {}

    # planes: prefer FM; fallback to MT props if present
    s1 = fm_p.get("nodal-plane-1-strike") or mt_p.get("nodal-plane-1-strike")
    d1 = fm_p.get("nodal-plane-1-dip")    or mt_p.get("nodal-plane-1-dip")
    r1 = (fm_p.get("nodal-plane-1-rake") or fm_p.get("nodal-plane-1-slip") or
          mt_p.get("nodal-plane-1-rake") or mt_p.get("nodal-plane-1-slip"))
    s2 = fm_p.get("nodal-plane-2-strike") or mt_p.get("nodal-plane-2-strike")
    d2 = fm_p.get("nodal-plane-2-dip")    or mt_p.get("nodal-plane-2-dip")
    r2 = (fm_p.get("nodal-plane-2-rake") or fm_p.get("nodal-plane-2-slip") or
          mt_p.get("nodal-plane-2-rake") or mt_p.get("nodal-plane-2-slip"))

    try:
        s1, d1, r1 = float(s1), float(d1), float(r1)
        s2, d2, r2 = float(s2), float(d2), float(r2)
    except (TypeError, ValueError):
        return None  # no numeric planes in products

    # axes (if available on MT)
    T_pl = mt_p.get("t-axis-plunge");  T_az = mt_p.get("t-axis-azimuth")
    N_pl = mt_p.get("n-axis-plunge");  N_az = mt_p.get("n-axis-azimuth")
    P_pl = mt_p.get("p-axis-plunge");  P_az = mt_p.get("p-axis-azimuth")
    axes = {}
    try:
        if None not in (T_pl, T_az, N_pl, N_az, P_pl, P_az):
            axes = {
                "T_plunge": float(T_pl), "T_azimuth": float(T_az),
                "N_plunge": float(N_pl), "N_azimuth": float(N_az),
                "P_plunge": float(P_pl), "P_azimuth": float(P_az),
            }
        else:
            axes = axes_from_sdr(s1, d1, r1)
    except Exception:
        axes = axes_from_sdr(s1, d1, r1)

    return {
        "strike1": s1, "dip1": d1, "rake1": r1,
        "strike2": s2, "dip2": d2, "rake2": r2,
        **axes,
        "source": (fm.get("source") if fm and fm.get("source") else
                   (mt.get("source") if mt else None) or "usgs")
    }

def _sdr_to_axes(strike: float, dip: float, rake: float):
    """
    Convert (strike, dip, rake) -> (T,N,P) axes as (azimuth, plunge) in degrees,
    using a unit double-couple in NED (North,East,Down). Conventions as GCMT.
    """
    st, di, ra = map(np.radians, [strike, dip, rake])

    # Fault normal (upper hemisphere) and slip, NED coordinates
    nN = -np.sin(di) * np.sin(st)
    nE =  np.sin(di) * np.cos(st)
    nD = -np.cos(di)
    sN =  np.cos(ra) * np.cos(st) + np.sin(ra) * np.cos(di) * np.sin(st)
    sE =  np.cos(ra) * np.sin(st) - np.sin(ra) * np.cos(di) * np.cos(st)
    sD =  np.sin(ra) * np.sin(di)

    n = np.array([nN, nE, nD]); n /= np.linalg.norm(n)
    s = np.array([sN, sE, sD]); s /= np.linalg.norm(s)

    # DC tensor (orientation only)
    M = np.outer(s, n) + np.outer(n, s)

    # Principal axes (ascending eigenvalues): P (min), N (mid), T (max)
    w, V = np.linalg.eigh(M)
    idx = np.argsort(w)
    V = V[:, idx]
    P, N, T = V[:, 0], V[:, 1], V[:, 2]

    def vec_to_az_pl(v):
        v = v / np.linalg.norm(v)
        # make plunge downward (Down >= 0)
        if v[2] < 0:
            v = -v
        Nn, Ee, Dd = v
        az = (math.degrees(math.atan2(Ee, Nn)) + 360.0) % 360.0
        pl = math.degrees(math.asin(max(-1.0, min(1.0, Dd))))
        return az, pl

    T_az, T_pl = vec_to_az_pl(T)
    N_az, N_pl = vec_to_az_pl(N)
    P_az, P_pl = vec_to_az_pl(P)
    return (T_az, T_pl), (N_az, N_pl), (P_az, P_pl)

def tnp_from_obspy_fm(fm):
    """
    Given an ObsPy FocalMechanism, return a dict matching your GCMT fields:
    T_plunge, T_azimuth, N_plunge, N_azimuth, P_plunge, P_azimuth.

    Priority:
      1) use fm.principal_axes if present
      2) else compute from nodal_plane_1 (strike,dip,rake)
      3) else (optionally) compute from fm.moment_tensor.tensor if you have it
    """
    # 1) Direct from principal_axes (if provided in QuakeML)
    pax = getattr(fm, "principal_axes", None)
    if pax and pax.t_axis and pax.n_axis and pax.p_axis:
        T_pl = float(pax.t_axis.plunge); T_az = float(pax.t_axis.azimuth) % 360.0
        N_pl = float(pax.n_axis.plunge); N_az = float(pax.n_axis.azimuth) % 360.0
        P_pl = float(pax.p_axis.plunge); P_az = float(pax.p_axis.azimuth) % 360.0
        return {
            "T_plunge": T_pl, "T_azimuth": T_az,
            "N_plunge": N_pl, "N_azimuth": N_az,
            "P_plunge": P_pl, "P_azimuth": P_az,
        }

    # 2) Compute from a nodal plane (use plane 1; plane 2 yields the same axes)
    npanes = getattr(fm, "nodal_planes", None)
    if npanes and npanes.nodal_plane_1:
        s = float(npanes.nodal_plane_1.strike)
        d = float(npanes.nodal_plane_1.dip)
        r = float(npanes.nodal_plane_1.rake)
        (T_az, T_pl), (N_az, N_pl), (P_az, P_pl) = _sdr_to_axes(s, d, r)
        return {
            "T_plunge": T_pl, "T_azimuth": T_az,
            "N_plunge": N_pl, "N_azimuth": N_az,
            "P_plunge": P_pl, "P_azimuth": P_az,
        }


if __name__ == '__main__':

    input_file = 'global_2/anss.csv'
    output_file_prefix = 'global_2/anss_mechanisms'

    client = Client("USGS")
    client.help()


    hypo_cat = pandas.read_csv(input_file)

    fieldnames = [
        "id", "time_iso", "longitude", "latitude", "depth", "mag", "mag_type",
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
        "T_plunge", "T_azimuth", "N_plunge", "N_azimuth", "P_plunge", "P_azimuth", "source"
    ]

    catalog = []
    for i, event in list(hypo_cat.iterrows())[1688:1689]:
        focal_catalog = client.get_events(eventid=event.id)
        print(f'Processing event {i}')
        print(focal_catalog)

        if focal_catalog[0].focal_mechanisms:
            pref_fm = None

            for fm in focal_catalog[0].focal_mechanisms:
                if fm.nodal_planes is not None:
                    pref_fm = focal_catalog[0].focal_mechanisms[0]
                    break
            if pref_fm.nodal_planes is None:
                print(f'Event {i},  id: {event.id} has no focal nodal planes')
                continue

            np1 = [pref_fm.nodal_planes.nodal_plane_1.strike,
                   pref_fm.nodal_planes.nodal_plane_1.dip,
                   pref_fm.nodal_planes.nodal_plane_1.rake]
            np2 = [pref_fm.nodal_planes.nodal_plane_1.strike,
                   pref_fm.nodal_planes.nodal_plane_1.dip,
                   pref_fm.nodal_planes.nodal_plane_1.rake]

            principal_axes = tnp_from_obspy_fm(pref_fm)

            parsed_event = [event.id,
                            event.time,
                            event.longitude, event.latitude,
                            event.depth,
                            event.mag,
                            event.magType,
                            *np1, *np2,
                            principal_axes['T_plunge'],
                            principal_axes['T_azimuth'],
                            principal_axes['N_plunge'],
                            principal_axes['N_azimuth'],
                            principal_axes['P_plunge'],
                            principal_axes['P_azimuth'],
                            "usgs"
                            ]
            catalog.append(parsed_event)



        elif True:
            # >>> Fallback: query USGS API GeoJSON detail and extract planes/axes
            try:
                products = fetch_products_from_comcat(str(event.id))
                print(products)
                mech = extract_mech_from_products(products)
            except Exception as e:
                mech = None
                print(f"  USGS API fallback failed for {event.id}: {e}")

            if mech:
                np1 = [mech["strike1"], mech["dip1"], mech["rake1"]]
                np2 = [mech["strike2"], mech["dip2"], mech["rake2"]]

                # use API axes if present; else compute from plane 1
                if all(k in mech and mech[k] is not None for k in
                       ("T_plunge", "T_azimuth", "N_plunge", "N_azimuth", "P_plunge", "P_azimuth")):
                    Tpl, Taz = mech["T_plunge"], mech["T_azimuth"]
                    Npl, Naz = mech["N_plunge"], mech["N_azimuth"]
                    Ppl, Paz = mech["P_plunge"], mech["P_azimuth"]
                else:
                    axes = axes_from_sdr(*np1)
                    Tpl, Taz = axes["T_plunge"], axes["T_azimuth"]
                    Npl, Naz = axes["N_plunge"], axes["N_azimuth"]
                    Ppl, Paz = axes["P_plunge"], axes["P_azimuth"]

                parsed_event = [event.id,
                                event.time,
                                event.longitude, event.latitude,
                                event.depth,
                                event.mag,
                                event.magType,
                                *np1, *np2,
                                Tpl, Taz, Npl, Naz, Ppl, Paz,
                                mech.get("source", "usgs")]
                catalog.append(parsed_event)
            else:
                print(f"Event {i}, id={event.id}: no mechanism via ObsPy or USGS API")

        else:
            print(f'Event {i},  id: {event.id} has no focal mechanisms')


    mechanism_catalog = pandas.DataFrame(data=catalog, columns=fieldnames)
    mechanism_catalog = mechanism_catalog.sort_values(
        by="time_iso",
        key=lambda s: pandas.to_datetime(s, utc=True, errors="coerce"),
        na_position="last"
    )
    mechanism_catalog.to_csv(output_file_prefix + f'_test.csv', sep=',', index=False)