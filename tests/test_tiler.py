import mercantile
import pytest
from rasterio.crs import CRS
from shapely.geometry import Polygon

from aiocogeo_tiler import COGTiler

INFILE = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
INFILE_NODATA = (
    "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_nodata.tif"
)


@pytest.mark.asyncio
async def test_cog_tiler_tile():
    async with COGTiler(INFILE) as cog:
        centroid = Polygon.from_bounds(*cog.bounds).centroid
        tile = await cog.tile(
            *mercantile.tile(centroid.x, centroid.y, cog.maxzoom),
            tilesize=256,
            resampling_method="bilinear"
        )

        assert tile.data.shape == (3, 256, 256)
        assert tile.data.dtype == cog.profile["dtype"]


@pytest.mark.asyncio
async def test_cog_tiler_point():
    async with COGTiler(INFILE) as cog:
        centroid = Polygon.from_bounds(*cog.bounds).centroid
        val = await cog.point(lon=centroid.x, lat=centroid.y)
        assert list(val) == [50, 69, 74]


@pytest.mark.asyncio
async def test_cog_tiler_part():
    async with COGTiler(INFILE_NODATA) as cog:
        tile = await cog.part(
            bbox=(-10526706.9, 4445561.5, -10526084.1, 4446144.0),
            bbox_crs=CRS.from_epsg(cog.epsg),
        )
        assert tile.data.shape == (3, 976, 1043)
        assert tile.mask is not None


@pytest.mark.asyncio
async def test_cog_tiler_part_dimensions():
    async with COGTiler(INFILE_NODATA) as cog:
        tile = await cog.part(
            bbox=(-10526706.9, 4445561.5, -10526084.1, 4446144.0),
            bbox_crs=CRS.from_epsg(cog.epsg),
            width=500,
            height=500,
        )

        assert tile.data.shape == (3, 500, 500)


@pytest.mark.asyncio
async def test_cog_tiler_preview():
    async with COGTiler(INFILE) as cog:
        tile = await cog.preview()
        assert tile.data.shape == (3, 1024, 864)


@pytest.mark.asyncio
async def test_cog_tiler_preview_max_size():
    async with COGTiler(INFILE) as cog:
        tile = await cog.preview(max_size=412)
        assert tile.data.shape == (3, 412, 348)


@pytest.mark.asyncio
async def test_cog_tiler_preview_dimensions():
    async with COGTiler(INFILE) as cog:
        tile = await cog.preview(width=512, height=512)
        assert tile.data.shape == (3, 512, 512)


@pytest.mark.asyncio
async def test_cog_tiler_info():
    async with COGTiler(INFILE) as cog:
        info = await cog.info()
        assert info["minzoom"] == cog.minzoom == 11
        assert info["maxzoom"] == cog.maxzoom == 17
        assert info["colorinterp"] == ["red", "green", "blue"]


@pytest.mark.asyncio
async def test_cog_tiler_stats():
    async with COGTiler(INFILE) as cog:
        stats = await cog.stats()
        assert stats["0"].percentiles == [25, 208]
        assert stats["1"].percentiles == [37, 214]
        assert stats["2"].percentiles == [48, 214]
