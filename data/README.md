# data/

Scripts to download and preprocess all datasets.

## Files

| File | Purpose |
|------|---------|
| `prepare_xbd.py` | Download + tile xBD dataset, generate damage masks |
| `prepare_bright.py` | Download + tile BRIGHT dataset (optical + SAR) |
| `download_sentinel.py` | Download Sentinel-1/2 imagery for target AOIs via Copernicus |
| `preprocess_sentinel.py` | Cloud-filter, reproject, generate pre/post stacks |

## Data sources

- **xBD**: https://xview2.org/dataset
- **BRIGHT**: https://zenodo.org/record/BRIGHT (Hugging Face mirror available)
- **Sentinel-1/2**: https://dataspace.copernicus.eu
- **ACLED**: https://acleddata.com
