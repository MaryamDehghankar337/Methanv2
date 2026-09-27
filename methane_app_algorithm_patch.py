"""
Algorithm replacement for the Streamlit Sentinel-2 methane app.

How to use:
1. Open the supplied app file (paste.txt) and replace the old functions
   normalized_difference through run_algorithm with the functions below.
2. Keep the UI, STAC search, Process API download and output code unchanged.
3. The app can continue using Sentinel-2 L1C data. Because L1C has no SCL,
   this patch constructs conservative cloud/shadow/water/edge masks from the
   available bands and uses the target/reference intersection.

Important limitation:
- L1C cannot provide a true SCL cloud mask or true surface reflectance.
- The algorithm therefore reports methane *candidates*, not validated flux.
- It intentionally does not calculate an emission rate from L1C.
"""

import warnings
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing,
    gaussian_filter,
    label,
    generate_binary_structure,
    maximum_filter,
)


# Add these parameters to the existing PARAMS dictionary.
ALGORITHM_PARAMS = {
    "reference_min_corr": 0.55,
    "reference_max_median_shift": 0.20,
    "reference_max_mad_ratio": 2.5,
    "edge_margin_pixels": 8,
    "cloud_bright_quantile": 0.995,
    "cloud_ndsi": 0.40,
    "cloud_ndwi": 0.35,
    "water_mndwi": 0.12,
    "water_ndwi": 0.28,
    "vegetation_ndvi": 0.55,
    "use_vegetation_mask": False,
    "lrad_b03_quantile": 0.03,
    "background_sigma": 30.0,
    "smoothing_sigma": 0.8,
    "threshold_z": 3.0,
    "absolute_floor": 0.0008,
    "minimum_component_pixels": 12,
    "maximum_component_pixels": 2500,
    "maximum_scene_candidate_fraction": 0.01,
    "final_dilation": 1,
    "max_source_distance_pixels": 150,
}


def _finite_stack(bands):
    return np.logical_and.reduce([
        np.isfinite(bands[band]) for band in BANDS
    ])


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = (
        np.isfinite(first) & np.isfinite(second) &
        (np.abs(denominator) > 1e-8)
    )
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def _robust_median_mad(array, valid):
    values = array[valid & np.isfinite(array)]
    if values.size == 0:
        return np.nan, np.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-8)


def _scene_features(bands):
    b03 = bands["B03"]
    b04 = bands["B04"]
    b08 = bands["B08"]
    b11 = bands["B11"]
    b12 = bands["B12"]
    return {
        "ndvi": normalized_difference(b08, b04),
        "ndwi": normalized_difference(b03, b08),
        "mndwi": normalized_difference(b03, b11),
        "ndbi": normalized_difference(b11, b08),
        "ndsi": normalized_difference(b03, b11),
        "swir_sum": b11 + b12,
    }


def _edge_valid(shape, margin):
    mask = np.ones(shape, dtype=bool)
    if margin <= 0:
        return mask
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def build_l1c_quality_mask(bands, q_value=None):
    """Conservative L1C quality mask using only B03/B04/B08/B11/B12."""
    shape = bands["B11"].shape
    finite = _finite_stack(bands)
    features = _scene_features(bands)

    reflectance_valid = np.ones(shape, dtype=bool)
    for name in BANDS:
        values = bands[name]
        # Process API reflectance should be [0, 1]. Allow a small margin.
        reflectance_valid &= np.isfinite(values) & (values >= 0.0) & (values <= 1.2)

    # No-data/outside-swath pixels in Process API outputs often become zero.
    reflectance_valid &= (bands["B03"] > 1e-5) & (bands["B11"] > 1e-5)

    # Water and wet surfaces: use both MNDWI and NDWI to reduce false positives.
    water = (
        (features["mndwi"] > ALGORITHM_PARAMS["water_mndwi"]) |
        (features["ndwi"] > ALGORITHM_PARAMS["water_ndwi"])
    )

    # Bright cloud/cirrus proxy. This is not SCL; it is deliberately conservative.
    brightness = np.nanmedian(
        np.stack([bands["B03"], bands["B04"], bands["B08"]]), axis=0
    )
    bright_limit = np.nanquantile(
        brightness[np.isfinite(brightness)],
        ALGORITHM_PARAMS["cloud_bright_quantile"]
    ) if np.isfinite(brightness).any() else np.inf
    cloud_proxy = brightness >= bright_limit
    cloud_proxy |= features["ndsi"] > ALGORITHM_PARAMS["cloud_ndsi"]
    cloud_proxy |= features["ndwi"] > ALGORITHM_PARAMS["cloud_ndwi"]

    # Vegetation can be retained for arid scenes; it is optional because
    # vegetation is a surface type, not automatically an invalid pixel.
    vegetation = features["ndvi"] > ALGORITHM_PARAMS["vegetation_ndvi"]

    # Saturated or extreme SWIR pixels.
    saturation = (bands["B11"] >= 1.0) | (bands["B12"] >= 1.0)

    valid = (
        finite & reflectance_valid &
        ~water & ~cloud_proxy & ~saturation &
        _edge_valid(shape, ALGORITHM_PARAMS["edge_margin_pixels"])
    )
    if ALGORITHM_PARAMS["use_vegetation_mask"]:
        valid &= ~vegetation

    if q_value is not None:
        valid &= bands["B03"] > float(q_value)

    diagnostics = {
        "water_pixels": int(water.sum()),
        "cloud_proxy_pixels": int(cloud_proxy.sum()),
        "vegetation_pixels": int(vegetation.sum()),
        "saturation_pixels": int(saturation.sum()),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
    }
    return valid, diagnostics, features


def _stable_pair_mask(target, reference):
    target_features = _scene_features(target)
    reference_features = _scene_features(reference)

    valid = _finite_stack(target) & _finite_stack(reference)
    for name in BANDS:
        valid &= (
            np.isfinite(target[name]) & np.isfinite(reference[name]) &
            (target[name] > 0) & (reference[name] > 0) &
            (target[name] < 1.2) & (reference[name] < 1.2)
        )

    # Reject pixels with extreme target/reference reflectance change before
    # fitting c. This suppresses fields, water edges and moving cloud remnants.
    for name in ("B03", "B04", "B08", "B11", "B12"):
        ratio = np.abs(target[name] - reference[name]) / np.maximum(reference[name], 0.02)
        valid &= ratio < 0.75

    # Both dates must pass the L1C proxy mask.
    target_mask, target_diag, _ = build_l1c_quality_mask(target)
    reference_mask, reference_diag, _ = build_l1c_quality_mask(reference)
    valid &= target_mask & reference_mask

    return valid, target_diag, reference_diag


def calculate_lrad(bands, q_value):
    """Compatibility wrapper: returns valid pixels, not artifact pixels."""
    valid, _, _ = build_l1c_quality_mask(bands, q_value=q_value)
    return valid


def calculate_c(b11, b12, valid):
    use = valid & np.isfinite(b11) & np.isfinite(b12)
    use &= (b11 > 0.02) & (b12 > 0.02) & (b11 < 1.0) & (b12 < 1.0)
    if int(use.sum()) < 100:
        return 1.0
    x = b12[use].astype(np.float64)
    y = b11[use].astype(np.float64)
    denominator = np.sum(x * x)
    return float(np.sum(x * y) / max(denominator, 1e-20))


def calculate_mbsp(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (np.abs(b12) > 1e-8)
    output[use] = ((c * b12[use] - b11[use]) / b12[use]).astype(np.float32)
    return output


def _remove_background(array, valid):
    data = np.where(valid & np.isfinite(array), array, 0.0).astype(np.float32)
    weights = (valid & np.isfinite(array)).astype(np.float32)
    sigma = ALGORITHM_PARAMS["background_sigma"]
    smooth_data = gaussian_filter(data, sigma=sigma)
    smooth_weights = gaussian_filter(weights, sigma=sigma)
    background = np.zeros_like(array, dtype=np.float32)
    good = smooth_weights > 0.05
    background[good] = smooth_data[good] / smooth_weights[good]
    residual = array - background
    residual[~valid] = np.nan
    return residual.astype(np.float32), background


def _robust_threshold(array, valid):
    median, robust_sigma = _robust_median_mad(array, valid)
    threshold = max(
        median + ALGORITHM_PARAMS["threshold_z"] * robust_sigma,
        ALGORITHM_PARAMS["absolute_floor"],
    )
    return threshold, median, robust_sigma


def _component_cleanup(binary, valid):
    structure = generate_binary_structure(2, 2)
    binary = binary_opening(binary, structure=structure)
    binary = binary_closing(binary, structure=structure)
    labels, count = label(binary, structure=structure)
    if count == 0:
        return np.zeros_like(binary), 0

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    min_size = int(ALGORITHM_PARAMS["minimum_component_pixels"])
    max_size = int(ALGORITHM_PARAMS["maximum_component_pixels"])
    retained = np.where((sizes >= min_size) & (sizes <= max_size))[0]
    retained = retained[retained != 0]
    cleaned = np.isin(labels, retained) & valid
    return cleaned, int(retained.size)


def run_algorithm(target, reference):
    """Run conservative L1C relative MBMP candidate screening.

    Returned arrays retain the keys expected by the current Streamlit UI.
    The 'final' array is a candidate mask only; no flux is inferred.
    """
    target_q = float(np.nanquantile(target["B03"], ALGORITHM_PARAMS["l1c_b03_quantile"]))
    reference_q = float(np.nanquantile(reference["B03"], ALGORITHM_PARAMS["l1c_b03_quantile"]))

    common_valid, target_diag, reference_diag = _stable_pair_mask(target, reference)
    common_valid &= target["B03"] > target_q
    common_valid &= reference["B03"] > reference_q

    if int(common_valid.sum()) < 100:
        raise RuntimeError(
            "Too few stable pixels remain after L1C quality masking. "
            "Try a clearer target/reference pair or a smaller AOI."
        )

    c_target = calculate_c(target["B11"], target["B12"], common_valid)
    c_reference = calculate_c(reference["B11"], reference["B12"], common_valid)
    target_mbsp = calculate_mbsp(target["B11"], target["B12"], c_target, common_valid)
    reference_mbsp = calculate_mbsp(reference["B11"], reference["B12"], c_reference, common_valid)

    relative = target_mbsp - reference_mbsp
    relative[~common_valid] = np.nan

    residual, background = _remove_background(relative, common_valid)
    smooth = gaussian_filter(
        np.where(common_valid & np.isfinite(residual), residual, 0.0),
        sigma=ALGORITHM_PARAMS["smoothing_sigma"],
    )
    smooth[~common_valid] = np.nan

    threshold, median, robust_sigma = _robust_threshold(smooth, common_valid)
    initial = common_valid & np.isfinite(smooth) & (smooth > threshold)
    final, regions = _component_cleanup(initial, common_valid)

    if ALGORITHM_PARAMS["final_dilation"] > 0:
        final = binary_dilation(
            final,
            iterations=int(ALGORITHM_PARAMS["final_dilation"]),
        ) & common_valid

    # Do not allow a scene-wide noisy mask to be called a plume.
    candidate_fraction = float(final.sum() / max(common_valid.sum(), 1))
    if candidate_fraction > ALGORITHM_PARAMS["maximum_scene_candidate_fraction"]:
        warnings.warn(
            "Candidate fraction is too large; scene is likely dominated by "
            "surface/reference artifacts. Final mask was cleared."
        )
        final[:] = False
        regions = 0

    # UI-compatible result dictionary plus additional diagnostics.
    return {
        "relative": relative,
        "gaussian": smooth,
        "background": background,
        "residual": residual,
        "valid": common_valid,
        "initial": initial,
        "connected": final,
        "final": final,
        "mean": float(median),
        "std": float(robust_sigma),
        "threshold": float(threshold),
        "regions": int(regions),
        "valid_count": int(common_valid.sum()),
        "initial_count": int(initial.sum()),
        "final_count": int(final.sum()),
        "candidate_fraction": candidate_fraction,
        "c_target": float(c_target),
        "c_reference": float(c_reference),
        "target_quality": target_diag,
        "reference_quality": reference_diag,
        "screening_only": True,
        "flux_available": False,
    }
