"""Estimate SRAM bitcell, array, and macro area from configured geometry."""


def _normalize_sram_cell_type(cell_type: str) -> str:
    raw = str(cell_type).strip().upper()
    if raw in {"6T", "SRAM_6T", "SRAM_6T_CELL"}:
        return "SRAM_6T_CELL"
    if raw in {"10T", "SRAM_10T", "SRAM_10T_CELL"}:
        return "SRAM_10T_CELL"
    raise ValueError(f"Unsupported cell_type: {cell_type}")


def estimate_bitcell_area(
    # Transistor dimensions (m)
    w_access: float,       # Access NMOS / PG width
    w_pd: float,           # Pull-down NMOS width
    w_pu: float,           # Pull-up PMOS width
    l_transistor: float,   # Transistor length (kept for interface consistency)

    # FreePDK45 Design Rules (default values in meters)
    CPP: float = 0.18e-6,          # Contacted Poly Pitch
    MMP: float = 0.16e-6,          # Minimum Metal Pitch
    diffusion_spacing: float = 0.08e-6,
    gate_contact_spacing: float = 0.05e-6,
    metal_overhang: float = 0.03e-6,

    *,
    w_fd=None,                     # 10T only: feed-forward NMOS width
    cell_type: str = "SRAM_6T_CELL",
) -> float:
    """
    Estimate SRAM bitcell area for 6T / 10T using the same coarse geometry idea.

    6T:
        Keep the existing estimation logic unchanged.

    10T:
        Reuse the 6T methodology, but treat the 10T cell as:
        - the original 6T storage core
        - plus one extra feedback / stacked pull-down branch

        So compared with 6T, the width model gains one extra branch term.
        Height is kept the same as the 6T estimator to preserve the original style.
    """
    cell_type = _normalize_sram_cell_type(cell_type)

    w_access = float(w_access)
    w_pd = float(w_pd)
    w_pu = float(w_pu)
    _ = float(l_transistor)  # currently not used in the geometric area model

    # Shared terms
    access_width = w_access + 2 * gate_contact_spacing
    pd_pu_width = max(w_pd, w_pu) + 2 * diffusion_spacing

    if cell_type == "SRAM_6T_CELL":
        # Keep old 6T logic exactly as before
        width_columns = 3 * CPP  # 2 access + 1 inverter pair
        width_scaling = (access_width + pd_pu_width) / (0.135e-6 + 0.205e-6)
        cell_width = width_columns * width_scaling + 2 * metal_overhang
        cell_height = 2 * MMP + 2 * diffusion_spacing + 4 * metal_overhang
        return cell_width * cell_height

    # 10T branch
    if w_fd is None:
        raise ValueError("w_fd must be provided when cell_type='SRAM_10T_CELL'")

    w_fd = float(w_fd)

    # 10T approximation:
    # - original access branch
    # - original storage-core branch
    # - one extra FD / stacked-PD helper branch
    fd_branch_width = max(w_pd, w_fd) + 2 * diffusion_spacing

    # Compared with 6T's 3-column abstraction, 10T is modeled as one extra branch.
    width_columns = 4 * CPP

    # Reference widths for current 10T default sizing style:
    # pg=135nm, core(max(pd, pu))=90nm, helper(max(pd, fd))=135nm
    width_scaling = (
        access_width + pd_pu_width + fd_branch_width
    ) / (0.135e-6 + 0.09e-6 + 0.135e-6)

    cell_width = width_columns * width_scaling + 2 * metal_overhang

    # Keep the same height model as 6T to preserve the original estimation philosophy.
    cell_height = 2 * MMP + 2 * diffusion_spacing + 4 * metal_overhang

    return cell_width * cell_height

def estimate_total_area(num_rows, num_cols, num_arrays):
    """
    估算多阵列 SRAM 总面积 (µm²)。
    使用 OpenRAM FreePDK45nm 单阵列面积回归公式乘以阵列数。
    """
    return estimate_array_area(num_cols, num_rows) * num_arrays


def estimate_array_area(word_size, num_words):
    width = 31.3547 + word_size * 2.9918 + num_words * 0.0568
    height = 49.9556 + word_size * 1.0801 + num_words * 0.0830
    return width * height

def estimate_array_macro_area(
    num_rows,
    num_cols,
    current_bitcell_area,
    default_bitcell_area,
):
    num_cells = num_rows * num_cols
    base_array_area = estimate_array_area(num_cols, num_rows)
    base_cell_area = default_bitcell_area * num_cells
    periphery_area = base_array_area - base_cell_area

    if periphery_area < 0:
        raise ValueError(
            f"Computed periphery_area is negative: {periphery_area}. "
            "Area models are inconsistent for this architecture."
        )

    return periphery_area + current_bitcell_area * num_cells


def estimate_total_macro_area(
    num_rows,
    num_cols,
    num_arrays,
    current_bitcell_area,
    default_bitcell_area,
):
    array_area = estimate_array_macro_area(
        num_rows=num_rows,
        num_cols=num_cols,
        current_bitcell_area=current_bitcell_area,
        default_bitcell_area=default_bitcell_area,
    )
    return array_area * num_arrays

_DR_CPP = 0.18e-6
_DR_MMP = 0.16e-6
_DR_DIFF_SP = 0.08e-6
_DR_GATE_CT = 0.05e-6
_DR_CONTACT = 0.065e-6
_DR_METAL_OH = 0.03e-6

_REF_PG = 0.135e-6
_REF_PD = 0.205e-6
_REF_PU = 0.090e-6
_REF_FD = 0.135e-6
_REF_L = 50e-9


def _cell_width_metric(
    w_access: float,
    w_pd: float,
    w_pu: float,
    l_transistor: float,
    *,
    cell_type: str = "SRAM_6T_CELL",
    w_fd=None,
) -> float:
    cell_type = _normalize_sram_cell_type(cell_type)

    w_access = float(w_access)
    w_pd = float(w_pd)
    w_pu = float(w_pu)
    l_transistor = float(l_transistor)

    active_pg = w_access + 2 * _DR_GATE_CT
    active_inv = w_pd + w_pu + 2 * _DR_DIFF_SP
    cpp = l_transistor + 2 * _DR_GATE_CT

    if cell_type == "SRAM_6T_CELL":
        return active_pg + active_inv + 3 * cpp

    if w_fd is None:
        raise ValueError("w_fd must be provided when cell_type='SRAM_10T_CELL'")

    w_fd = float(w_fd)
    active_helper = max(w_pd, w_fd) + 2 * _DR_DIFF_SP

    return active_pg + active_inv + active_helper + 3 * cpp


def _cell_height_metric(
    l_transistor: float,
    *,
    cell_type: str = "SRAM_6T_CELL",
) -> float:
    cell_type = _normalize_sram_cell_type(cell_type)
    l_transistor = float(l_transistor)

    gate_pitch = l_transistor + _DR_GATE_CT + _DR_CONTACT

    if cell_type == "SRAM_6T_CELL":
        return 3 * gate_pitch + 2 * _DR_MMP + 2 * _DR_METAL_OH

    return 4 * gate_pitch + 3 * _DR_MMP + 2 * _DR_METAL_OH


_REF_CW_6T = _cell_width_metric(
    _REF_PG,
    _REF_PD,
    _REF_PU,
    _REF_L,
    cell_type="SRAM_6T_CELL",
)
_REF_CH_6T = _cell_height_metric(
    _REF_L,
    cell_type="SRAM_6T_CELL",
)

_REF_CW_10T = _cell_width_metric(
    _REF_PG,
    _REF_PD,
    _REF_PU,
    _REF_L,
    cell_type="SRAM_10T_CELL",
    w_fd=_REF_FD,
)
_REF_CH_10T = _cell_height_metric(
    _REF_L,
    cell_type="SRAM_10T_CELL",
)

_BASE_COL_PITCH_6T = 1.20
_BASE_ROW_PITCH_6T = 0.90

# 用默认 10T 几何尺寸相对默认 6T 的比例，生成 10T 的基准 pitch
_BASE_COL_PITCH_10T = _BASE_COL_PITCH_6T * (_REF_CW_10T / _REF_CW_6T)
_BASE_ROW_PITCH_10T = _BASE_ROW_PITCH_6T * (_REF_CH_10T / _REF_CH_6T)


def estimate_scaled_array_area(
    num_rows: int,
    num_cols: int,
    num_arrays: int,
    w_access: float,
    w_pd: float,
    w_pu: float,
    l_transistor: float,
    sa_max_width: float = 0.54e-6,
    wld_max_width: float = 0.27e-6,
    prc_max_width: float = 0.27e-6,
    *,
    cell_type: str = "SRAM_6T_CELL",
    w_fd=None,
) -> float:
    """
    Estimate total SRAM area in m^2 using the pitch-scaling model.
    Supports both 6T and 10T.
    """
    import math

    cell_type = _normalize_sram_cell_type(cell_type)

    if cell_type == "SRAM_10T_CELL" and w_fd is None:
        raise ValueError("w_fd must be provided when cell_type='SRAM_10T_CELL'")

    if cell_type == "SRAM_10T_CELL":
        ref_cw = _REF_CW_10T
        ref_ch = _REF_CH_10T
        base_col_pitch = _BASE_COL_PITCH_10T
        base_row_pitch = _BASE_ROW_PITCH_10T
    else:
        ref_cw = _REF_CW_6T
        ref_ch = _REF_CH_6T
        base_col_pitch = _BASE_COL_PITCH_6T
        base_row_pitch = _BASE_ROW_PITCH_6T

    w_ratio = _cell_width_metric(
        w_access,
        w_pd,
        w_pu,
        l_transistor,
        cell_type=cell_type,
        w_fd=w_fd,
    ) / ref_cw

    h_ratio = _cell_height_metric(
        l_transistor,
        cell_type=cell_type,
    ) / ref_ch

    col_pitch = base_col_pitch * w_ratio
    row_pitch = base_row_pitch * h_ratio

    ref_wld_w = 0.27e-6
    row_log_factor = 0.60 + 0.40 * math.log2(max(num_rows, 2)) / math.log2(32)
    wld_r = max(float(wld_max_width), 1e-9) / ref_wld_w
    overhead_x = (9.59 + 4.11 * wld_r) * row_log_factor

    ref_sa_w = 0.54e-6
    ref_prc_w = 0.27e-6
    sa_r = max(float(sa_max_width), 1e-9) / ref_sa_w
    prc_r = max(float(prc_max_width), 1e-9) / ref_prc_w
    overhead_y = 5.77 + 5.13 * sa_r + 1.92 * prc_r

    width_um = overhead_x + num_cols * col_pitch
    height_um = overhead_y + num_rows * row_pitch

    return width_um * height_um * num_arrays * 1e-12
