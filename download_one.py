import lightkurve as lk

tic_id = 307210830  # WASP-100b — confirmed planet with nice transits

search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
print("Available sectors:")
print(search)

# Download first available sector
lc = search[0].download()
lc.to_fits(f"TIC_{tic_id}_lightcurve.fits", overwrite=True)
print(f"\nSaved: TIC_{tic_id}_lightcurve.fits")
print(f"Columns: {list(lc.columns)}")

