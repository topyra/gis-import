Python 3.6+
pip install numpy pillow

Usage:
1) Download set of adjacent .asc heightmap tiles to one directory
2) Merge all files using ersi_asc_merge.py to one .asc file
3) Using ersi_asc_to_ue_png.py convert .asc file to .png heightmap
   a) By default map is resized to default UE5 map segment sizes
5) Import heightmap to Unreal Engine Editor as landscape, or any other engine


Geoportal maps:
https://mapy.geoportal.gov.pl/imapnext/imap/index.html
Panel narzędzi -> zawartość map -> dane do pobrania -> numeryczny model terenu -> siatka <=1m -> NMT* (use available for your localization)

Tested on 0.5m grid maps (but should work on different grid sizes).

Not tested:
- partially overlapping heightmaps
- holes between heightmap (also small gaps between segments)
 
