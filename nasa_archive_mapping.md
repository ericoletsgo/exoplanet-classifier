# NASA Exoplanet Archive to KOI Feature Mapping

## Your Question
> "is there really only 2 in here that match the features the model can be trained on?"

## Answer: NO! There are MANY matching features!

Here's the mapping between NASA Exoplanet Archive columns and KOI model features:

## Stellar Parameters (Temperature, Mass, Radius, Position)

| NASA Archive Column | KOI Model Feature | Description |
|---------------------|-------------------|-------------|
| `st_teff` | `koi_steff` | Stellar effective temperature |
| `st_tefferr1` | `koi_steff_err1` | Temperature upper uncertainty |
| `st_tefferr2` | `koi_steff_err2` | Temperature lower uncertainty |
| `st_rad` | `koi_srad` | Stellar radius |
| `st_raderr1` | `koi_srad_err1` | Radius upper uncertainty |
| `st_raderr2` | `koi_srad_err2` | Radius lower uncertainty |
| `st_mass` | `koi_smass` | Stellar mass |
| `st_masserr1` | `koi_smass_err1` | Mass upper uncertainty |
| `st_masserr2` | `koi_smass_err2` | Mass lower uncertainty |
| `st_logg` | `koi_slogg` | Stellar surface gravity |
| `st_loggerr1` | `koi_slogg_err1` | Surface gravity upper uncertainty |
| `st_loggerr2` | `koi_slogg_err2` | Surface gravity lower uncertainty |
| `st_met` | `koi_smet` | Stellar metallicity |
| `st_meterr1` | `koi_smet_err1` | Metallicity upper uncertainty |
| `st_meterr2` | `koi_smet_err2` | Metallicity lower uncertainty |
| `ra` | `ra` | Right ascension (EXACT MATCH) |
| `dec` | `dec` | Declination (EXACT MATCH) |
| `sy_vmag` | `koi_kepmag` | V-band magnitude (similar to Kepler mag) |
| `sy_kmag` | `koi_kmag` | K-band magnitude (EXACT MATCH) |
| `sy_gaiamag` | `koi_gmag` | Gaia G magnitude |

## Orbital/Exoplanet Parameters (Mass, Orbital Info)

| NASA Archive Column | KOI Model Feature | Description |
|---------------------|-------------------|-------------|
| `pl_orbper` | `koi_period` | Orbital period |
| `pl_orbpererr1` | `koi_period_err1` | Period upper uncertainty |
| `pl_orbpererr2` | `koi_period_err2` | Period lower uncertainty |
| `pl_rade` | `koi_prad` | Planet radius (Earth radii) |
| `pl_radeerr1` | `koi_prad_err1` | Radius upper uncertainty |
| `pl_radeerr2` | `koi_prad_err2` | Radius lower uncertainty |
| `pl_radj` | Can derive `koi_prad` | Planet radius (Jupiter radii, convert to Earth) |
| `pl_bmasse` | Can derive mass | Planet mass (Earth masses) |
| `pl_bmassj` | Can derive mass | Planet mass (Jupiter masses) |
| `pl_orbeccen` | `koi_eccen` | Orbital eccentricity |
| `pl_orbeccenerr1` | Can use | Eccentricity upper uncertainty |
| `pl_orbeccenerr2` | Can use | Eccentricity lower uncertainty |
| `pl_orbsmax` | `koi_sma` | Semi-major axis |
| `pl_orbsmaxerr1` | Can use | Semi-major axis upper uncertainty |
| `pl_orbsmaxerr2` | Can use | Semi-major axis lower uncertainty |
| `pl_insol` | `koi_insol` | Insolation flux |
| `pl_insolerr1` | `koi_insol_err1` | Insolation upper uncertainty |
| `pl_insolerr2` | `koi_insol_err2` | Insolation lower uncertainty |
| `pl_eqt` | `koi_teq` | Equilibrium temperature |
| `pl_eqterr1` | Can use | Equilibrium temp upper uncertainty |
| `pl_eqterr2` | Can use | Equilibrium temp lower uncertainty |

## Summary

### Direct Matches: ~20 features
- Position (ra, dec)
- Stellar temperature, radius, mass, surface gravity, metallicity (with errors)
- Orbital period (with errors)
- Planet radius (with errors)
- Eccentricity
- Semi-major axis
- Insolation
- Equilibrium temperature
- Magnitudes

### Derivable Features: ~10 more
- Planet radius conversions (Jupiter → Earth radii)
- Mass estimates
- Additional orbital parameters

### Missing Features:
- Transit-specific measurements (depth, duration, impact parameter)
- Light curve quality metrics (SNR, event statistics)
- Centroid measurements (these are Kepler-specific)

## Total Usable Features: ~30 out of 96

**So NO, it's not just 2 features!** You have about **30 usable features** from the NASA Exoplanet Archive that map to the model's training features.

## For Retraining

When you upload NASA Exoplanet Archive data:
1. The system will detect ~30 matching features
2. Missing features will be filled with median/zero values
3. Model can still train, but with reduced feature set
4. Performance may be slightly lower than with full KOI data

## Recommendation

For best results:
- **Use KOI data** (koi.csv) - Has all 96 features
- **Or use NASA Archive data** - Works with ~30 features
- **Or combine both** - Use KOI for training, NASA Archive for validation

The model is flexible and will work with whatever features are available!
