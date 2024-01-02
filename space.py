from pyproj import CRS

class Space:
    pass

class GeoSpace(Space):
    def __init__(self, crs) -> None:
        self.crs = crs
        if not CRS(crs).is_projected:
            raise ValueError("The CRS should be projected (metric).")
