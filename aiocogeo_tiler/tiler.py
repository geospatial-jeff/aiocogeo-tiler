"""aiocogeo tiler"""
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import morecantile
import numpy as np
from aiocogeo.cog import COGReader
from morecantile import TileMatrixSet
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.rio.overview import get_maximum_overview_level
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.warp import transform as transform_coords
from rasterio.warp import transform_bounds
from rio_tiler.io import AsyncBaseReader
from rio_tiler.models import ImageData, ImageStatistics, Info
from rio_tiler.utils import _stats as raster_stats

# Rasterio <-> PIL resampling modes
RESAMPLING = {
    "nearest": 0,
    "bilinear": 2,
    "cubic": 3,
    "lanczos": 1,
}


DEFAULT_TMS = morecantile.tms.get("WebMercatorQuad")
WGS84 = CRS.from_epsg(4326)


@dataclass
class COGInfo:
    """cog info response"""

    bounds: Tuple[float, float, float, float]
    center: Tuple[float, float, int]
    minzoom: int
    maxzoom: int
    band_metadata: List[Tuple[int, Dict]]
    band_descriptions: List[Tuple[int, str]]
    dtype: str
    colorinterp: List[str]
    nodata_type: str
    colormap: Optional[Dict[int, Sequence[int]]] = None


class TileResponse(ImageData):
    """tile response"""

    @classmethod
    def from_array(cls, array: Union[np.ndarray, np.ma.masked_array]) -> "TileResponse":
        """create from numpy array"""
        if isinstance(array, np.ma.masked_array):
            return cls(data=array.data, mask=(array.mask * 255).astype("uint8"))
        return cls(data=array)


@dataclass
class COGTiler(COGReader, AsyncBaseReader):
    """cogeo tiler"""

    tms: TileMatrixSet = field(default=DEFAULT_TMS)

    async def open(self):
        """override default open method"""
        await self._open()
        self.bounds = transform_bounds(
            CRS.from_epsg(self.epsg), CRS.from_epsg(4326), *self.native_bounds
        )
        self.minzoom, self.maxzoom = self.get_zooms()
        return self

    def get_zooms(self, tilesize: int = 256) -> Tuple[int, int]:
        """Calculate raster min/max zoom level."""
        dst_affine, w, h = calculate_default_transform(
            CRS.from_epsg(self.epsg),
            self.tms.crs,
            self.profile["width"],
            self.profile["height"],
            *self.native_bounds,
        )
        resolution = max(abs(dst_affine[0]), abs(dst_affine[4]))
        maxzoom = self.tms.zoom_for_res(resolution)

        overview_level = get_maximum_overview_level(w, h, minsize=tilesize)
        ovr_resolution = resolution * (2 ** overview_level)
        minzoom = self.tms.zoom_for_res(ovr_resolution)

        return minzoom, maxzoom

    async def _warped_read(
        self,
        bounds: Tuple[int, int, int, int],
        width: int,
        height: int,
        bounds_crs: CRS,
        resample_method: int = Image.NEAREST,
    ) -> np.ndarray:
        """partial read with crs handling"""
        src_transform = from_bounds(*bounds, width=width, height=height)
        bounds = transform_bounds(bounds_crs, CRS.from_epsg(self.epsg), *bounds)
        dst_transform = from_bounds(*bounds, width=width, height=height)
        arr = await self.read(
            bounds, shape=(width, height), resample_method=resample_method
        )
        arr, _ = reproject(
            arr,
            destination=np.empty((self.profile["count"], width, height)),
            src_transform=dst_transform,
            dst_transform=src_transform,
            src_crs=CRS.from_epsg(self.epsg),
            dst_crs=bounds_crs,
        )
        return arr.astype(self.profile["dtype"])

    async def info(self) -> Info:
        """info"""
        if self.has_alpha:
            nodata_type = "Alpha"
        elif self.is_masked:
            nodata_type = "Mask"
        elif self.nodata is not None:
            nodata_type = "Nodata"
        else:
            nodata_type = "None"

        # TODO: Figure out where scale, offset, band_metadata, and band_descriptions come from
        band_descr = [(ix, f"band{ix}") for ix in self.indexes]
        band_meta = [(ix, {}) for ix in self.indexes]  # type:ignore

        return Info(
            bounds=self.bounds,
            center=self.center,
            minzoom=self.minzoom,
            maxzoom=self.maxzoom,
            band_metadata=band_meta,
            band_descriptions=band_descr,
            dtype=self.profile["dtype"],
            nodata_type=nodata_type,
            colorinterp=[color.name for color in self.color_interp],
            colormap=self.colormap,
        )

    async def stats(
        self,
        pmin: float = 2.0,
        pmax: float = 98.0,
        hist_options: Optional[Dict] = None,
        indexes: Optional[Union[Sequence[int], int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_size: int = 1024,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        bounds_crs: CRS = CRS.from_epsg(4326),
        resampling_method: str = "nearest",
    ) -> Dict[str, ImageStatistics]:
        """stats"""

        hist_options = hist_options or {}

        if self.colormap and not "bins" not in hist_options:
            hist_options["bins"] = [
                k for k, v in self.colormap.items() if v != (0, 0, 0, 255)
            ]

        if isinstance(indexes, int):
            indexes = (indexes,)

        if indexes is None:
            indexes = [
                idx for idx, b in enumerate(self.color_interp) if b != ColorInterp.alpha
            ]
            if len(indexes) != self.profile["count"]:
                warnings.warn("Alpha band was removed from the output data array")
            indexes = range(self.profile["count"])

        if bounds:
            resp = await self.part(
                bounds,
                bbox_crs=bounds_crs,
                width=width,
                height=height,
                resampling_method=resampling_method,
            )
        else:
            resp = await self.preview(
                width=width,
                height=height,
                max_size=max_size,
                resampling_method=resampling_method,
            )

        data = np.ma.array(resp.data)

        return {
            str(indexes[b]): ImageStatistics.parse_obj(
                raster_stats(data[b], percentiles=(pmin, pmax), **hist_options)
            )
            for b in range(data.shape[0])
        }

    async def tile(
        self,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        tilesize: int = 256,
        indexes: Optional[Union[int, Sequence]] = None,
        expression: Optional[str] = "",  # placeholder
        resampling_method: str = "nearest",
    ) -> TileResponse:
        """tile"""

        if isinstance(indexes, int):
            indexes = (indexes,)

        tile = morecantile.Tile(x=tile_x, y=tile_y, z=tile_z)
        tile_bounds = self.tms.xy_bounds(tile)

        # resampling_method -> resample_method
        resample_method = RESAMPLING[resampling_method]

        width = height = tilesize
        if self.epsg != self.tms.crs.to_epsg():
            arr = await self._warped_read(
                tile_bounds,
                width,
                height,
                bounds_crs=self.tms.crs,
                resample_method=resample_method,
            )
        else:
            arr = await self.read(
                tile_bounds, shape=(width, height), resample_method=resample_method
            )

        # should be handled at read level
        if indexes:
            arr = arr[[r - 1 for r in indexes]]

        return TileResponse.from_array(arr)

    async def part(
        self,
        bbox: Tuple[float, float, float, float],
        bbox_crs: CRS = WGS84,
        width: int = None,
        height: int = None,
        resampling_method: str = "nearest",
    ) -> TileResponse:
        """part"""
        if bbox_crs != CRS.from_epsg(self.epsg):
            bbox = transform_bounds(bbox_crs, CRS.from_epsg(self.epsg), *bbox)

        if not height or not width:
            width = math.ceil((bbox[2] - bbox[0]) / self.profile["transform"].a)
            height = math.ceil((bbox[3] - bbox[1]) / -self.profile["transform"].e)

        # resampling_method -> resample_method
        resample_method = RESAMPLING[resampling_method]

        arr = await self.read(
            bounds=bbox, shape=(width, height), resample_method=resample_method
        )
        return TileResponse.from_array(arr)

    async def preview(
        self,
        width: int = None,
        height: int = None,
        max_size: int = 1024,
        resampling_method: str = "nearest",
    ) -> TileResponse:
        """preview"""
        # https://github.com/cogeotiff/rio-tiler/blob/master/rio_tiler/reader.py#L293-L303
        if not height and not width:
            if max(self.profile["height"], self.profile["width"]) < max_size:
                height, width = self.profile["height"], self.profile["width"]
            else:
                ratio = self.profile["height"] / self.profile["width"]
                if ratio > 1:
                    height = max_size
                    width = math.ceil(height / ratio)
                else:
                    width = max_size
                    height = math.ceil(width * ratio)

        # resampling_method -> resample_method
        resample_method = RESAMPLING[resampling_method]

        arr = await self.read(
            bounds=self.native_bounds,
            shape=(width, height),
            resample_method=resample_method,
        )
        return TileResponse.from_array(arr)

    async def point(self, lon: float, lat: float, **kwargs: Any) -> List:
        """point"""
        coords = [lon, lat]
        if self.epsg != 4326:
            coords = [
                pt[0]
                for pt in transform_coords(
                    WGS84, CRS.from_epsg(self.epsg), [coords[0]], [coords[1]]
                )
            ]
        ifd = self.ifds[0]
        geotransform = self.geotransform()
        invgt = ~geotransform

        # Transform request point to pixel coordinates relative to geotransform
        image_x, image_y = invgt * coords
        xtile = math.floor((image_x + 1e-6) / ifd.TileWidth.value)
        ytile = math.floor((image_y + 1e-6) / ifd.TileHeight.value)
        tile = await self.get_tile(xtile, ytile, 0)

        # Calculate index of pixel relative to the tile
        xindex = int(image_x % ifd.TileWidth.value)
        yindex = int(image_y % ifd.TileHeight.value)

        return tile[:, xindex, yindex].tolist()
