def float_to_dtype_array(
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],
) -> np.ndarray:
    """Convert a float array back to the original image dtype without bias."""

    arr = np.clip(arr, 0.0, 1.0)
    np_dtype = np.dtype(dtype)

    if np.issubdtype(np_dtype, np.integer):
        dtype_info = np.iinfo(np_dtype)
        dtype_max = float(dtype_info.max)
        arr_out = np.rint(arr * dtype_max).astype(np_dtype)
        if alpha is not None:
            alpha_out = np.rint(np.clip(alpha, 0.0, 1.0) * dtype_max).astype(np_dtype)
            alpha_out = alpha_out[:, :, None]
            arr_out = np.concatenate([arr_out, alpha_out], axis=2)
    elif np.issubdtype(np_dtype, np.floating):
        arr_out = arr.astype(np_dtype, copy=False)
        if alpha is not None:
            alpha_out = np.clip(alpha, 0.0, 1.0).astype(np_dtype, copy=False)
            alpha_out = alpha_out[:, :, None]
            arr_out = np.concatenate([arr_out, alpha_out], axis=2)
    else:
        # For uncommon dtypes, fall back to float32 to avoid surprises.
        arr_out = arr.astype(np.float32)
        if alpha is not None:
            alpha_out = np.clip(alpha, 0.0, 1.0).astype(np.float32)
            alpha_out = alpha_out[:, :, None]
            arr_out = np.concatenate([arr_out, alpha_out], axis=2)

    return np.ascontiguousarray(arr_out)