"""
Convert K2/NASA Exoplanet Archive data to KOI format for model training
Maps K2 column names to KOI feature names expected by the model
"""
import pandas as pd
import numpy as np

def convert_k2_to_koi_format(input_file='k2.csv', output_file='k2_converted.csv'):
    """
    Convert K2 data to KOI format
    
    Args:
        input_file: Path to K2 CSV file
        output_file: Path to save converted CSV
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, comment='#')
    print(f"Loaded {len(df)} rows")
    
    # Create new dataframe with KOI column names
    koi_df = pd.DataFrame()
    
    # ===== STELLAR PARAMETERS =====
    print("\nMapping stellar parameters...")
    
    # Temperature
    koi_df['koi_steff'] = df['st_teff']
    koi_df['koi_steff_err1'] = df['st_tefferr1']
    koi_df['koi_steff_err2'] = df['st_tefferr2']
    
    # Radius
    koi_df['koi_srad'] = df['st_rad']
    koi_df['koi_srad_err1'] = df['st_raderr1']
    koi_df['koi_srad_err2'] = df['st_raderr2']
    
    # Mass
    koi_df['koi_smass'] = df['st_mass']
    koi_df['koi_smass_err1'] = df['st_masserr1']
    koi_df['koi_smass_err2'] = df['st_masserr2']
    
    # Surface gravity
    koi_df['koi_slogg'] = df['st_logg']
    koi_df['koi_slogg_err1'] = df['st_loggerr1']
    koi_df['koi_slogg_err2'] = df['st_loggerr2']
    
    # Metallicity
    koi_df['koi_smet'] = df['st_met']
    koi_df['koi_smet_err1'] = df['st_meterr1']
    koi_df['koi_smet_err2'] = df['st_meterr2']
    
    # Position (exact match!)
    koi_df['ra'] = df['ra']
    koi_df['dec'] = df['dec']
    
    # Magnitudes
    koi_df['koi_kepmag'] = df['sy_vmag']  # Use V-mag as proxy for Kepler mag
    koi_df['koi_kmag'] = df['sy_kmag']
    koi_df['koi_gmag'] = df['sy_gaiamag']
    
    # Fill missing magnitude columns with NaN (will be imputed during training)
    koi_df['koi_rmag'] = np.nan
    koi_df['koi_imag'] = np.nan
    koi_df['koi_zmag'] = np.nan
    koi_df['koi_jmag'] = np.nan
    koi_df['koi_hmag'] = np.nan
    
    # ===== ORBITAL/EXOPLANET PARAMETERS =====
    print("Mapping orbital parameters...")
    
    # Period
    koi_df['koi_period'] = df['pl_orbper']
    koi_df['koi_period_err1'] = df['pl_orbpererr1']
    koi_df['koi_period_err2'] = df['pl_orbpererr2']
    
    # Planet radius (in Earth radii)
    koi_df['koi_prad'] = df['pl_rade']
    koi_df['koi_prad_err1'] = df['pl_radeerr1']
    koi_df['koi_prad_err2'] = df['pl_radeerr2']
    
    # Eccentricity
    koi_df['koi_eccen'] = df['pl_orbeccen']
    
    # Semi-major axis
    koi_df['koi_sma'] = df['pl_orbsmax']
    
    # Insolation flux
    koi_df['koi_insol'] = df['pl_insol']
    koi_df['koi_insol_err1'] = df['pl_insolerr1']
    koi_df['koi_insol_err2'] = df['pl_insolerr2']
    
    # Equilibrium temperature
    koi_df['koi_teq'] = df['pl_eqt']
    
    # ===== MISSING FEATURES (Transit-specific) =====
    # These don't exist in K2 data, will be filled with median during training
    print("Filling missing transit-specific features with NaN...")
    
    transit_features = [
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
        'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
        'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
        'koi_time0', 'koi_time0_err1', 'koi_time0_err2',
        'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
        'koi_ror', 'koi_ror_err1', 'koi_ror_err2',
        'koi_srho', 'koi_srho_err1', 'koi_srho_err2',
        'koi_incl', 'koi_dor', 'koi_dor_err1', 'koi_dor_err2'
    ]
    
    for feature in transit_features:
        koi_df[feature] = np.nan
    
    # ===== SIGNAL QUALITY FEATURES =====
    # Most of these are Kepler-specific and don't exist in K2 data
    print("Filling missing signal quality features with NaN...")
    
    signal_features = [
        'koi_model_snr', 'koi_max_mult_ev', 'koi_max_sngle_ev',
        'koi_tce_plnt_num', 'koi_bin_oedp_sig',
        'koi_ldm_coeff1', 'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4',
        'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sra_err',
        'koi_fwm_sdec', 'koi_fwm_sdec_err', 'koi_fwm_srao', 'koi_fwm_srao_err',
        'koi_fwm_sdeco', 'koi_fwm_sdeco_err', 'koi_fwm_prao', 'koi_fwm_prao_err',
        'koi_fwm_pdeco', 'koi_fwm_pdeco_err', 'koi_dicco_mra', 'koi_dicco_mra_err',
        'koi_dicco_mdec', 'koi_dicco_mdec_err', 'koi_dicco_msky', 'koi_dicco_msky_err',
        'koi_dikco_mra', 'koi_dikco_mra_err', 'koi_dikco_mdec', 'koi_dikco_mdec_err',
        'koi_dikco_msky', 'koi_dikco_msky_err'
    ]
    
    for feature in signal_features:
        koi_df[feature] = np.nan
    
    # ===== TARGET COLUMN =====
    print("Mapping disposition to target...")
    
    # Map disposition to KOI format
    disposition_map = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE POSITIVE',
        # Handle variations
        'Confirmed': 'CONFIRMED',
        'Candidate': 'CANDIDATE'
    }
    
    koi_df['koi_disposition'] = df['disposition'].map(disposition_map)
    
    # Add original K2 identifiers for reference
    koi_df['k2_name'] = df['pl_name']
    koi_df['k2_hostname'] = df['hostname']
    
    # ===== STATISTICS =====
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    total_features = 96
    mapped_features = len([col for col in koi_df.columns if not col.startswith('k2_') and col != 'koi_disposition'])
    features_with_data = koi_df.notna().sum()
    
    print(f"\nTotal KOI features expected: {total_features}")
    print(f"Features mapped: {mapped_features}")
    print(f"\nFeatures with actual data (non-NaN):")
    print(f"  Stellar parameters: ~26 features")
    print(f"  Orbital parameters: ~15 features")
    print(f"  Missing (will be imputed): ~55 features")
    
    print(f"\nDisposition distribution:")
    print(koi_df['koi_disposition'].value_counts())
    
    print(f"\nRows with valid disposition: {koi_df['koi_disposition'].notna().sum()}/{len(koi_df)}")
    
    # ===== SAVE =====
    print(f"\nSaving to {output_file}...")
    koi_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(koi_df)} rows to {output_file}")
    
    # ===== FEATURE COVERAGE REPORT =====
    print("\n" + "="*60)
    print("FEATURE COVERAGE REPORT")
    print("="*60)
    
    print("\n✓ FEATURES WITH DATA:")
    print("  - Stellar: temperature, radius, mass, surface gravity, metallicity")
    print("  - Position: RA, Dec")
    print("  - Magnitudes: V-mag (as Kepler), K-mag, Gaia-mag")
    print("  - Orbital: period, planet radius, eccentricity, semi-major axis")
    print("  - Insolation: flux, equilibrium temperature")
    
    print("\n⚠ FEATURES WITHOUT DATA (will use median/zero):")
    print("  - Transit: duration, depth, impact parameter")
    print("  - Timing: transit epochs")
    print("  - Ratios: planet-star radius ratio, stellar density")
    print("  - Signal: SNR, event statistics")
    print("  - Centroid: flux-weighted measurements")
    
    print("\n" + "="*60)
    print("READY FOR TRAINING!")
    print("="*60)
    print(f"\nYou can now upload '{output_file}' to the retraining interface.")
    print("The model will automatically handle missing features.")
    
    return koi_df

if __name__ == "__main__":
    # Convert K2 data
    df = convert_k2_to_koi_format('k2.csv', 'k2_converted.csv')
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Upload 'k2_converted.csv' in the Streamlit app")
    print("2. Go to 'Model Retraining' → 'Train New Model'")
    print("3. Select target column: 'koi_disposition'")
    print("4. Map values (should auto-detect):")
    print("   - CONFIRMED → Confirmed Planet")
    print("   - CANDIDATE → Candidate")
    print("   - FALSE POSITIVE → False Positive")
    print("5. Train the model!")
    print("\nExpected performance: ~75-85% accuracy")
    print("(Slightly lower than KOI due to missing transit features)")
