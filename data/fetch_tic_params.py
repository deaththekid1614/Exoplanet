from astroquery.mast import Catalogs
import pandas as pd
import time

tic_ids = [
    150428135, 410153553, 38890006, 382862991, 219134,
    307210830, 2733520, 2441619, 183120439, 217897806,
    98796344, 149603524, 268301217, 155867025, 144065872,
    281541555, 300843387, 167418143, 198008474, 33521996,
    140691463
]

results = []
for tic_id in tic_ids:
    try:
        print(f"Querying TIC {tic_id}...")
        result = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
        if len(result) > 0:
            row = result[0]
            
            # Astropy Row uses dict-style access, not .get()
            teff = float(row['teff']) if 'teff' in row.columns and row['teff'] is not None else 5778.0
            logg = float(row['logg']) if 'logg' in row.columns and row['logg'] is not None else 4.44
            radius = float(row['rad']) if 'rad' in row.columns and row['rad'] is not None else 1.0
            mass = float(row['mass']) if 'mass' in row.columns and row['mass'] is not None else 1.0
            gaiaid = int(row['gaiaid']) if 'gaiaid' in row.columns and row['gaiaid'] is not None else None
            
            # BP-RP from Gaia BP and RP magnitudes
            bp = row['gaiabp'] if 'gaiabp' in row.columns else None
            rp = row['gaiarp'] if 'gaiarp' in row.columns else None
            if bp is not None and rp is not None:
                bp_rp = float(bp - rp)
            else:
                bp_rp = 0.82
            
            results.append({
                "ID": tic_id,
                "TEFF": teff,
                "LOGG": logg,
                "RADIUS": radius,
                "MASS": mass,
                "BP_RP": bp_rp,
                "GAIA3ID": gaiaid,
            })
            print(f"  ✓ Got data: R={radius:.2f} Rsun, Teff={teff:.0f}K")
        else:
            print(f"  ✗ No data for TIC {tic_id}")
        time.sleep(0.3)
    except Exception as e:
        print(f"  ✗ Error for TIC {tic_id}: {e}")

df = pd.DataFrame(results)
df.to_csv("tic_catalog.csv", index=False)
print(f"\nSaved {len(df)} rows to tic_catalog.csv")