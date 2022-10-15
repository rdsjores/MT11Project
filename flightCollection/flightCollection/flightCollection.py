from datetime import timedelta

import numpy as np
import pandas as pd


def Conv_LongLat_to_Lambart_93(lon_deg, lat_deg):
    """function returning lambert coodonate with longitude and latitude"""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)

    lambda_0 = np.radians(3)

    phi_0 = np.radians(46.5)
    phi_1 = np.radians(44)
    phi_2 = np.radians(49)

    x_0 = 700000
    y_0 = 6600000

    R_T = 6371000

    n = np.log(np.cos(phi_1) / np.cos(phi_2)) / np.log(
        np.tan(0.25 * np.pi + 0.5 * phi_2) / np.tan(0.25 * np.pi + 0.5 * phi_1)
    )

    F = (
        R_T
        * np.cos(phi_1)
        * np.exp(n * np.log(np.tan(np.pi / 4 + phi_1 / 2)))
        / n
    )
    rho_0 = F / np.exp(n * np.log(np.tan(np.pi / 4 + phi_0 / 2)))
    rho = F / np.exp(n * np.log(np.tan(np.pi / 4 + lat / 2)))

    x = x_0 + rho * np.sin(n * (lon - lambda_0))
    y = y_0 + rho_0 - rho * np.cos(n * (lon - lambda_0))

    print(n)
    print(F)
    print(rho_0)

    return [x, y]


def iterate_time(data, threshold):
    """yield data by contigus timestamp"""
    idx = np.where(data.timestamp.diff().dt.total_seconds() > threshold)[0]
    start = 0
    for stop in idx:
        yield data.iloc[start:stop]
        start = stop + 1
    yield data.iloc[start:]


def iterate_icao24_callsign(data):
    """yield by flight icao and callsign"""
    for _, chunk in data.groupby(["icao24", "callsign"]):
        yield chunk


class FlightCollection:
    """class containing a dataframes of flight data and methods to clean, filter and display them"""

    def __init__(self, data):
        """default constructor with dataframe in argin"""
        self.data = data

    def __repr__(self) -> str:
        """default str convertor with shape info"""
        return f"FlightCollection with {self.data.shape[0]} records\n"

    @classmethod
    def read_json(cls, path):
        """constructor with json path in argin"""
        df = pd.read_json(path)
        return FlightCollection(df)

    def __iter__(self):
        """default iterator override by yield over flight callsign and icao"""
        for group in iterate_icao24_callsign(self.data):
            for flight in iterate_time(group, 20000):
                yield Flight(flight)

    def __len__(self):
        """default size getter retuning rows count"""
        return sum(1 for _ in self)

    def __getitem__(self, key):
        """getter by icao, call sign or date (period 1 day)"""
        if isinstance(key, str):
            result = FlightCollection(
                self.data.query("callsign == @key or icao24 == @key")
            )
        if isinstance(key, pd.Timestamp):
            before = key
            after = key + timedelta(days=1)
            result = FlightCollection(
                self.data.query("@before < timestamp < @after")
            )

        if len(result) == 1:
            return Flight(result.data)
        else:
            return result

    def clean_nan_line(self, column_name):
        """remove rows with Nan on the argin column name"""
        self.data = self.data.query(f"{column_name} == {column_name}")

    def born_long_lat(self, min_lon, max_lon, min_lat, max_lat):
        """remove row with coordonate out of interval"""
        print(f"{min_lat}< latitude <{max_lat} or latitude.isnull()")
        self.data = self.data.query(
            f"{min_lon} < longitude < {max_lon} or longitude.isnull()"
        )
        self.data = self.data.query(
            f"{min_lat} < latitude < {max_lat} or latitude.isnull()"
        )

    def remove_nan_column(self):
        """remove column with only nan values"""
        for c in self.data.columns:
            print(
                f"column : {c} has {self.data[c].count()} consistant values\n"
            )
            if self.data[c].count() == 0:
                self.data = self.data.drop([c], axis=1)

    def extract_low_altitude_flight(self, altitude_seuil):
        """return the data with an altitude under the threshold"""
        flights = self.data.groupby(["icao24", "callsign"])
        from_to_paris_flight = flights.filter(
            lambda x: x["altitude"].min() < altitude_seuil
        )
        from_to_paris_flight_collection = FlightCollection(from_to_paris_flight)
        return from_to_paris_flight_collection

    def add_Lambert_coord(self):
        """add the lambert 93 xy coordinate from the longitude and the latitude"""
        [x_lamb, y_lamb] = Conv_LongLat_to_Lambart_93(
            self.data["longitude"], self.data["latitude"]
        )
        self.data["x_lamb"] = x_lamb
        self.data["y_lamb"] = y_lamb


from cartopy.crs import EuroPP, PlateCarree


class Flight:
    """class contening a data frame of a particular flight and its methods to display it"""

    def __init__(self, data):
        """default constructor from a dataframe"""
        self.data = data

    def __repr__(self):
        """default str convertor with icao call sign and take off day"""
        return f"callsign: {min(self.data.callsign.values)},icao24: {min(self.data.icao24.values)}, day : {min(self.data.timestamp.values)}\n"

    def __lt__(self, other):
        """default comparator by timestamp"""
        return min(self.data.timestamp.values) <= min(
            other.data.timestamp.values
        )

    def plot(self, ax_in):
        """default displaying with latitude and longitude"""
        self.data.query("latitude==latitude").plot(
            ax=ax_in,
            x="longitude",
            y="latitude",
            legend=False,
            transform=PlateCarree(),
            color="crimson" if self.is_going_up() else "steelblue",
        )

    def plot_Lambert(self, ax_in):
        """display the lambert coordonate"""
        self.data.query("latitude==latitude").plot(
            ax=ax_in,
            x="x_lamb",
            y="y_lamb",
            legend=False,
            # transform=PlateCarree(),
            color="crimson" if self.is_going_up() else "steelblue",
        )

    def is_going_up(self):
        """return true if the flight is going up false if going down"""
        return self.data["vertical_rate"].mean(skipna=True) > 0
