import time
import numpy as np
import matplotlib.pyplot as plt

print("=" * 82)
print("NON-ABELIAN MASS-GENERATION CORE MECHANISM (Numerical Study)")
print("SU(2) lattice Yang-Mills: restricted-family evidence with DeltaV_int / DeltaV_NA")
print("=" * 82)

# =============================================================================
# SU(2) BASICS
# =============================================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
I2 = np.eye(2, dtype=np.complex64)

def su2_exp_np(omega):
    """
    Exponential map: omega in R^3 -> U in SU(2)
    U = cos(|omega|/2) I + i sin(|omega|/2) (n · sigma)
    """
    omega = np.asarray(omega, dtype=np.float32)
    norm = np.sqrt(np.sum(omega**2) + 1e-12)
    n = omega / norm
    cos_half = np.cos(norm / 2.0)
    sin_half = np.sin(norm / 2.0)
    n_dot_sigma = n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z
    return cos_half * I2 + 1j * sin_half * n_dot_sigma

def su2_alg_matrix(omega):
    """
    su(2) anti-Hermitian algebra element:
        A = i/2 * (omega · sigma)
    """
    return 0.5j * (
        omega[0] * SIGMA_X +
        omega[1] * SIGMA_Y +
        omega[2] * SIGMA_Z
    )

def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ], dtype=np.float32)

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float32)

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

def rotate_color_field(omega, R):
    return np.einsum("...a,ab->...b", omega, R).astype(np.float32)

def abelianize_projection(omega, axis=2):
    """
    U(1)-embedded baseline inside SU(2):
    keep only one color component.
    """
    out = np.zeros_like(omega)
    out[..., axis] = omega[..., axis]
    return out

# =============================================================================
# LATTICE HELPERS
# =============================================================================

def periodic_displacement(coord, center, L):
    """
    Shortest displacement on periodic lattice, allowing non-integer center.
    """
    return ((coord - center + L / 2.0) % L) - L / 2.0

def compute_plaquette_np(omega, L, x, y, z, t, mu, nu):
    def get_U(ix, iy, iz, it, idir):
        return su2_exp_np(omega[ix % L, iy % L, iz % L, it % L, idir])

    shift_mu = [1 if i == mu else 0 for i in range(4)]
    shift_nu = [1 if i == nu else 0 for i in range(4)]

    x_mu = (x + shift_mu[0]) % L
    y_mu = (y + shift_mu[1]) % L
    z_mu = (z + shift_mu[2]) % L
    t_mu = (t + shift_mu[3]) % L

    x_nu = (x + shift_nu[0]) % L
    y_nu = (y + shift_nu[1]) % L
    z_nu = (z + shift_nu[2]) % L
    t_nu = (t + shift_nu[3]) % L

    U1 = get_U(x, y, z, t, mu)
    U2 = get_U(x_mu, y_mu, z_mu, t_mu, nu)
    U3 = get_U(x_nu, y_nu, z_nu, t_nu, mu).conj().T
    U4 = get_U(x, y, z, t, nu).conj().T
    return U1 @ U2 @ U3 @ U4

def compute_wilson_action_and_vorticity(omega, L, beta=2.0):
    """
    Wilson action:
        S = beta * sum_p (1 - Re Tr P / 2)

    Plaquette-deviation functional:
        V = sum_p ||P - I||_F^2

    Exact SU(2) identity:
        S = (beta/4) * V
    """
    S_total = 0.0
    V_total = 0.0
    n_plaquettes = 0

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            P = compute_plaquette_np(omega, L, x, y, z, t, mu, nu)
                            tr_half = np.real(np.trace(P)) / 2.0
                            S_total += 1.0 - tr_half
                            V_total += np.sum(np.abs(P - I2) ** 2)
                            n_plaquettes += 1

    return beta * float(S_total), float(V_total), n_plaquettes

# =============================================================================
# FIELD CONSTRUCTION
# =============================================================================

def create_instanton_config(L, center, rho, charge=1):
    """
    BPST-like controlled lattice ansatz with possibly non-integer center.
    """
    omega = np.zeros((L, L, L, L, 4, 3), dtype=np.float32)
    cx, cy, cz, ct = center

    eta = np.zeros((3, 4, 4), dtype=np.float32)
    eta[0, 0, 1] = 1; eta[0, 1, 0] = -1
    eta[0, 2, 3] = 1; eta[0, 3, 2] = -1
    eta[1, 0, 2] = 1; eta[1, 2, 0] = -1
    eta[1, 3, 1] = 1; eta[1, 1, 3] = -1
    eta[2, 0, 3] = 1; eta[2, 3, 0] = -1
    eta[2, 1, 2] = 1; eta[2, 2, 1] = -1

    if charge == -1:
        eta = -eta

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    dx = periodic_displacement(x, cx, L)
                    dy = periodic_displacement(y, cy, L)
                    dz = periodic_displacement(z, cz, L)
                    dt = periodic_displacement(t, ct, L)

                    r = np.array([dx, dy, dz, dt], dtype=np.float32)
                    r2 = np.sum(r ** 2)
                    f = 2.0 * rho ** 2 / (r2 + rho ** 2 + 1e-6)

                    for mu in range(4):
                        for a in range(3):
                            val = 0.0
                            for nu in range(4):
                                val += eta[a, mu, nu] * r[nu]
                            omega[x, y, z, t, mu, a] = val * f / (rho ** 2 + 0.1)

    return omega

def create_two_lump_components(L, sep, rho=0.8, theta=0.0, rotation_axis="z"):
    """
    Topologically trivial two-lump family:
        omega_total = omega_+ + R(theta) omega_-
    with continuous centers to avoid sep-flooring artifacts.
    """
    center = L / 2.0
    center1 = (center - sep / 2.0, center, center, center)
    center2 = (center + sep / 2.0, center, center, center)

    omega1 = create_instanton_config(L, center1, rho, charge=+1)
    omega2 = create_instanton_config(L, center2, rho, charge=-1)

    if rotation_axis == "x":
        R = rotation_matrix_x(theta)
    elif rotation_axis == "y":
        R = rotation_matrix_y(theta)
    else:
        R = rotation_matrix_z(theta)

    omega2_rot = rotate_color_field(omega2, R)
    return omega1, omega2_rot

def create_two_lump_family(L, sep, rho=0.8, theta=0.0, rotation_axis="z", abelianized=False):
    omega1, omega2 = create_two_lump_components(L, sep, rho, theta, rotation_axis)

    if abelianized:
        omega1 = abelianize_projection(omega1, axis=2)
        omega2 = abelianize_projection(omega2, axis=2)

    return omega1 + omega2, omega1, omega2

def create_random_su2_field(L, scale=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(L, L, L, L, 4, 3)).astype(np.float32) * scale

def create_random_abelianized_field(L, scale=0.2, seed=0, axis=2):
    omega = create_random_su2_field(L, scale=scale, seed=seed)
    return abelianize_projection(omega, axis=axis)

# =============================================================================
# OVERLAP / EXCESS FUNCTIONALS
# =============================================================================

def commutator_overlap(omega1, omega2):
    """
    O = sum || [A1_mu, A2_nu] - [A1_nu, A2_mu] ||_F^2
    Vanishes in the commuting / Abelianized limit.
    """
    L = omega1.shape[0]
    total = 0.0

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            A1_mu = su2_alg_matrix(omega1[x, y, z, t, mu])
                            A1_nu = su2_alg_matrix(omega1[x, y, z, t, nu])
                            A2_mu = su2_alg_matrix(omega2[x, y, z, t, mu])
                            A2_nu = su2_alg_matrix(omega2[x, y, z, t, nu])

                            comm1 = A1_mu @ A2_nu - A2_nu @ A1_mu
                            comm2 = A1_nu @ A2_mu - A2_mu @ A1_nu
                            diff = comm1 - comm2
                            total += np.sum(np.abs(diff) ** 2)

    return float(total)

def relative_field_overlap(omega1, omega2):
    return float(np.sum(omega1 * omega2))

def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = np.mean(x)
    ym = np.mean(y)
    num = np.sum((x - xm) * (y - ym))
    den = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2) + 1e-12)
    return float(num / den)

def compute_family_metrics(L, sep, rho, theta, rotation_axis="z", beta=2.0):
    L = int(L)
    """
    Returns both non-Abelian and Abelianized metrics, including:
      DeltaV_int  = V_total - V1 - V2
      DeltaV_NA   = V_nonAb - V_Ab
      DeltaDeltaV_int = DeltaV_int_nonAb - DeltaV_int_Ab
    """
    # Non-Abelian family
    omega_tot_na, omega1_na, omega2_na = create_two_lump_family(
        L=L, sep=sep, rho=rho, theta=theta, rotation_axis=rotation_axis, abelianized=False
    )
    S_na, V_na, nplaq = compute_wilson_action_and_vorticity(omega_tot_na, L, beta=beta)
    _, V1_na, _ = compute_wilson_action_and_vorticity(omega1_na, L, beta=beta)
    _, V2_na, _ = compute_wilson_action_and_vorticity(omega2_na, L, beta=beta)
    O_na = commutator_overlap(omega1_na, omega2_na)

    DeltaV_int_na = V_na - V1_na - V2_na

    # Abelianized baseline
    omega_tot_ab, omega1_ab, omega2_ab = create_two_lump_family(
        L=L, sep=sep, rho=rho, theta=theta, rotation_axis=rotation_axis, abelianized=True
    )
    S_ab, V_ab, _ = compute_wilson_action_and_vorticity(omega_tot_ab, L, beta=beta)
    _, V1_ab, _ = compute_wilson_action_and_vorticity(omega1_ab, L, beta=beta)
    _, V2_ab, _ = compute_wilson_action_and_vorticity(omega2_ab, L, beta=beta)
    O_ab = commutator_overlap(omega1_ab, omega2_ab)

    DeltaV_int_ab = V_ab - V1_ab - V2_ab

    DeltaV_NA = V_na - V_ab
    DeltaDeltaV_int = DeltaV_int_na - DeltaV_int_ab

    return {
        "L": L,
        "sep": sep,
        "rho": rho,
        "theta": theta,
        "nplaq": nplaq,

        "S_nonAb": S_na,
        "V_nonAb": V_na,
        "V1_nonAb": V1_na,
        "V2_nonAb": V2_na,
        "O_nonAb": O_na,
        "DeltaV_int_nonAb": DeltaV_int_na,

        "S_Ab": S_ab,
        "V_Ab": V_ab,
        "V1_Ab": V1_ab,
        "V2_Ab": V2_ab,
        "O_Ab": O_ab,
        "DeltaV_int_Ab": DeltaV_int_ab,

        "DeltaV_NA": DeltaV_NA,
        "DeltaDeltaV_int": DeltaDeltaV_int,
    }

# =============================================================================
# EXPERIMENT A: EXPLICIT ABELIANIZED BASELINE
# =============================================================================

print("\n" + "=" * 82)
print("EXPERIMENT A: Explicit Abelianized baseline + DeltaV_NA / DeltaV_int")
print("=" * 82)

L = 6
beta = 2.0
theta_values_baseline = np.linspace(0.0, np.pi / 2.0, 6)
baseline_rows = []

print(
    f"{'theta':>8} {'V_nonAb':>12} {'V_Ab':>12} "
    f"{'DeltaV_NA':>12} {'O_nonAb':>12} {'O_Ab':>10} "
    f"{'DVint_NA':>12} {'DVint_Ab':>12}"
)
print("-" * 104)

for theta in theta_values_baseline:
    row = compute_family_metrics(L=L, sep=2.0, rho=0.8, theta=theta, rotation_axis="z", beta=beta)
    baseline_rows.append(row)

    print(
        f"{theta:8.4f} "
        f"{row['V_nonAb']:12.6f} {row['V_Ab']:12.6f} "
        f"{row['DeltaV_NA']:12.6f} {row['O_nonAb']:12.6f} {row['O_Ab']:10.6f} "
        f"{row['DeltaV_int_nonAb']:12.6f} {row['DeltaV_int_Ab']:12.6f}"
    )

print("\nInterpretation:")
print("- DeltaV_NA = V_nonAb - V_Ab isolates the excess tied to non-Abelian structure.")
print("- DeltaV_int measures interaction/excess beyond the sum of isolated lump contributions.")
print("- O_Ab ~ 0 confirms that the commutator channel is absent in the Abelianized baseline.")
print("- This remains a restricted-family mechanism study, not a spectral proof.")

# =============================================================================
# EXPERIMENT B: FINE THETA SCAN
# =============================================================================

print("\n" + "=" * 82)
print("EXPERIMENT B: Fine theta scan (30 points) with DeltaV_int / DeltaV_NA")
print("=" * 82)

theta_grid = np.linspace(0.0, np.pi, 30)
theta_scan_rows = []

print(
    f"{'idx':>4} {'theta':>10} {'V_nonAb':>12} {'V_Ab':>12} "
    f"{'DeltaV_NA':>12} {'DVint_NA':>12} {'DVint_Ab':>12} {'O_nonAb':>12}"
)
print("-" * 100)

for i, theta in enumerate(theta_grid):
    row = compute_family_metrics(L=L, sep=2.0, rho=0.8, theta=theta, rotation_axis="y", beta=beta)
    theta_scan_rows.append(row)

    print(
        f"{i:4d} {theta:10.6f} "
        f"{row['V_nonAb']:12.6f} {row['V_Ab']:12.6f} "
        f"{row['DeltaV_NA']:12.6f} "
        f"{row['DeltaV_int_nonAb']:12.6f} {row['DeltaV_int_Ab']:12.6f} "
        f"{row['O_nonAb']:12.6f}"
    )

V_nonAb_arr = np.array([r["V_nonAb"] for r in theta_scan_rows])
V_Ab_arr = np.array([r["V_Ab"] for r in theta_scan_rows])
O_nonAb_arr = np.array([r["O_nonAb"] for r in theta_scan_rows])
DeltaV_NA_arr = np.array([r["DeltaV_NA"] for r in theta_scan_rows])
DVint_NA_arr = np.array([r["DeltaV_int_nonAb"] for r in theta_scan_rows])
DVint_Ab_arr = np.array([r["DeltaV_int_Ab"] for r in theta_scan_rows])
DeltaDeltaV_int_arr = np.array([r["DeltaDeltaV_int"] for r in theta_scan_rows])

corr_O_DeltaVNA = pearson_corr(O_nonAb_arr, DeltaV_NA_arr)
corr_O_DVint = pearson_corr(O_nonAb_arr, DVint_NA_arr)
corr_DeltaVNA_DVint = pearson_corr(DeltaV_NA_arr, DVint_NA_arr)

print("\nCorrelation summary:")
print(f"Pearson corr[ O_nonAb(theta), DeltaV_NA(theta) ]        = {corr_O_DeltaVNA:.6f}")
print(f"Pearson corr[ O_nonAb(theta), DeltaV_int_nonAb(theta) ] = {corr_O_DVint:.6f}")
print(f"Pearson corr[ DeltaV_NA(theta), DeltaV_int_nonAb(theta) ] = {corr_DeltaVNA_DVint:.6f}")

print("\nInterpretation:")
print("- DeltaV_NA(theta) tracks the non-Abelian excess over the Abelianized baseline.")
print("- DeltaV_int_nonAb(theta) tracks interaction/excess within the non-Abelian family.")
print("- O_nonAb(theta) is best treated as a non-commuting overlap indicator,")
print("  not automatically as the sole strength measure of the excess channel.")

# =============================================================================
# EXPERIMENT C: RANDOM CONTROLS
# =============================================================================

print("\n" + "=" * 82)
print("EXPERIMENT C: Random controls for exact plaquette identity")
print("=" * 82)

random_scales = [0.05, 0.10, 0.20, 0.30]
random_trials = 5
random_rows = []

print(f"{'family':>14} {'scale':>8} {'trial':>6} {'S':>14} {'(beta/4)V':>14} {'abs.err':>12} {'V':>14}")
print("-" * 90)

for scale in random_scales:
    for trial in range(random_trials):
        seed_base = int(1000 * scale) + trial * 17

        # Full SU(2)
        omega_rand = create_random_su2_field(L=L, scale=scale, seed=seed_base)
        S_rand, V_rand, _ = compute_wilson_action_and_vorticity(omega_rand, L, beta=beta)
        exact_rand = (beta / 4.0) * V_rand
        err_rand = abs(S_rand - exact_rand)

        random_rows.append({
            "family": "random_SU2",
            "scale": scale,
            "trial": trial,
            "S": S_rand,
            "V": V_rand,
            "exact": exact_rand,
            "err": err_rand
        })

        print(f"{'random_SU2':>14} {scale:8.3f} {trial:6d} {S_rand:14.6f} {exact_rand:14.6f} {err_rand:12.3e} {V_rand:14.6f}")

        # U(1)-embedded
        omega_ab = create_random_abelianized_field(L=L, scale=scale, seed=seed_base + 1, axis=2)
        S_ab, V_ab, _ = compute_wilson_action_and_vorticity(omega_ab, L, beta=beta)
        exact_ab = (beta / 4.0) * V_ab
        err_ab = abs(S_ab - exact_ab)

        random_rows.append({
            "family": "random_U1embed",
            "scale": scale,
            "trial": trial,
            "S": S_ab,
            "V": V_ab,
            "exact": exact_ab,
            "err": err_ab
        })

        print(f"{'random_U1embed':>14} {scale:8.3f} {trial:6d} {S_ab:14.6f} {exact_ab:14.6f} {err_ab:12.3e} {V_ab:14.6f}")

max_random_err = max(r["err"] for r in random_rows)
print(f"\nMaximum identity error across random controls: {max_random_err:.6e}")

print("\nInterpretation:")
print("- The exact relation S = (beta/4) V is algebraic and survives beyond the two-lump family.")
print("- The non-Abelian mechanism therefore does NOT sit in the identity itself,")
print("  but in excess / interaction channels such as DeltaV_NA and DeltaV_int.")

# =============================================================================
# EXPERIMENT D: RESTRICTED-FAMILY LANDSCAPE
# =============================================================================

print("\n" + "=" * 82)
print("EXPERIMENT D: Restricted-family landscape with DeltaV_int / DeltaV_NA")
print("=" * 82)

family_rows = []

print(
    f"{'sep':>6} {'rho':>8} {'theta':>8} {'V_nonAb':>12} {'V_Ab':>12} "
    f"{'DeltaV_NA':>12} {'DVint_NA':>12} {'DVint_Ab':>12} {'DDVint':>12}"
)
print("-" * 110)

for sep in [2.0, 3.0, 4.0]:
    for rho in [0.5, 0.8, 1.0, 1.2]:
        for theta in [0.0, np.pi / 4.0, np.pi / 2.0]:
            row = compute_family_metrics(L=L, sep=sep, rho=rho, theta=theta, rotation_axis="z", beta=beta)
            family_rows.append(row)

            print(
                f"{sep:6.1f} {rho:8.3f} {theta:8.4f} "
                f"{row['V_nonAb']:12.6f} {row['V_Ab']:12.6f} "
                f"{row['DeltaV_NA']:12.6f} "
                f"{row['DeltaV_int_nonAb']:12.6f} {row['DeltaV_int_Ab']:12.6f} "
                f"{row['DeltaDeltaV_int']:12.6f}"
            )

V_family_min = min(r["V_nonAb"] for r in family_rows)
DeltaV_NA_min = min(r["DeltaV_NA"] for r in family_rows)
DeltaDeltaV_int_min = min(r["DeltaDeltaV_int"] for r in family_rows)

print(f"\nRestricted-family minimum V_nonAb        = {V_family_min:.6f}")
print(f"Restricted-family minimum DeltaV_NA      = {DeltaV_NA_min:.6f}")
print(f"Restricted-family minimum DeltaDeltaV_int = {DeltaDeltaV_int_min:.6f}")
print("These are family-level observations only, not global lower bounds.")

# =============================================================================
# EXPERIMENT E: FINITE-SIZE SURVEY
# =============================================================================

print("\n" + "=" * 82)
print("EXPERIMENT E: Finite-size survey (no continuum claim)")
print("=" * 82)

size_rows = []

print(
    f"{'L':>4} {'sep':>6} {'rho':>8} {'V_nonAb':>12} {'V_Ab':>12} "
    f"{'DeltaV_NA':>12} {'DVint_NA':>12} {'DDVint':>12}"
)
print("-" * 90)

for Ls in [4, 5, 6, 7, 8]:
    sep = max(2.0, Ls / 3.0)
    rho = max(0.6, 0.13 * Ls)

    row = compute_family_metrics(L=Ls, sep=sep, rho=rho, theta=np.pi / 4.0, rotation_axis="y", beta=beta)
    size_rows.append(row)

    print(
        f"{Ls:4d} {sep:6.2f} {rho:8.3f} "
        f"{row['V_nonAb']:12.6f} {row['V_Ab']:12.6f} "
        f"{row['DeltaV_NA']:12.6f} "
        f"{row['DeltaV_int_nonAb']:12.6f} {row['DeltaDeltaV_int']:12.6f}"
    )

print("\nInterpretation:")
print("- This probes finite-size robustness of the excess channels.")
print("- It is NOT a proof of RG invariance, continuum stability, or Hamiltonian gap.")

# =============================================================================
# FIGURES
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel 1: exact identity
ax = axes[0, 0]
x_su2 = [r["V"] for r in random_rows if r["family"] == "random_SU2"]
y_su2 = [r["S"] for r in random_rows if r["family"] == "random_SU2"]
x_u1 = [r["V"] for r in random_rows if r["family"] == "random_U1embed"]
y_u1 = [r["S"] for r in random_rows if r["family"] == "random_U1embed"]

ax.scatter(x_su2, y_su2, label="random SU(2)", alpha=0.7)
ax.scatter(x_u1, y_u1, label="random U(1)-embedded", alpha=0.7)
xmax = max(max(x_su2), max(x_u1)) * 1.05
xline = np.linspace(0.0, xmax, 100)
ax.plot(xline, (beta / 4.0) * xline, "r--", lw=2, label=f"S=(beta/4)V={(beta/4):.3f}V")
ax.set_title("Exact plaquette identity")
ax.set_xlabel("V")
ax.set_ylabel("S")
ax.grid(alpha=0.3)
ax.legend()

# Panel 2: theta scan V
ax = axes[0, 1]
theta_plot = np.array([r["theta"] for r in theta_scan_rows])
ax.plot(theta_plot, V_nonAb_arr, "o-", label="non-Abelian V")
ax.plot(theta_plot, V_Ab_arr, "s--", label="Abelianized V")
ax.set_title("Fine theta scan: V(theta)")
ax.set_xlabel("theta")
ax.set_ylabel("V")
ax.grid(alpha=0.3)
ax.legend()

# Panel 3: theta scan DeltaV_NA
ax = axes[0, 2]
ax.plot(theta_plot, DeltaV_NA_arr, "o-", label="DeltaV_NA")
ax.plot(theta_plot, DeltaDeltaV_int_arr, "s--", label="DeltaDeltaV_int")
ax.set_title("Fine theta scan: excess channels")
ax.set_xlabel("theta")
ax.set_ylabel("excess")
ax.grid(alpha=0.3)
ax.legend()

# Panel 4: theta scan DeltaV_int
ax = axes[1, 0]
ax.plot(theta_plot, DVint_NA_arr, "o-", label="DeltaV_int non-Ab")
ax.plot(theta_plot, DVint_Ab_arr, "s--", label="DeltaV_int Ab")
ax.set_title("Interaction excess: DeltaV_int(theta)")
ax.set_xlabel("theta")
ax.set_ylabel("DeltaV_int")
ax.grid(alpha=0.3)
ax.legend()

# Panel 5: overlap vs excess
ax = axes[1, 1]
ax.scatter(O_nonAb_arr, DeltaV_NA_arr, label="DeltaV_NA vs O", alpha=0.8)
ax.scatter(O_nonAb_arr, DVint_NA_arr, label="DeltaV_int non-Ab vs O", alpha=0.8)
ax.set_title("Overlap vs excess channels")
ax.set_xlabel("O_nonAb")
ax.set_ylabel("excess")
ax.grid(alpha=0.3)
ax.legend()

# Panel 6: finite-size survey
ax = axes[1, 2]
L_plot = [r["L"] for r in size_rows]
DeltaVNA_size = [r["DeltaV_NA"] for r in size_rows]
DDVint_size = [r["DeltaDeltaV_int"] for r in size_rows]
ax.plot(L_plot, DeltaVNA_size, "o-", label="DeltaV_NA")
ax.plot(L_plot, DDVint_size, "s--", label="DeltaDeltaV_int")
ax.set_title("Finite-size survey of excess channels")
ax.set_xlabel("L")
ax.set_ylabel("excess")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
out_path = "nonabelian_core_mechanism_with_deltas.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {out_path}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 82)
print("SUMMARY")
print("=" * 82)
print("1. The exact algebraic identity S = (beta/4) V is verified across")
print("   restricted two-lump families and random control fields.")
print("2. DeltaV_NA = V_nonAb - V_Ab isolates the non-Abelian excess over")
print("   the Abelianized baseline.")
print("3. DeltaV_int = V_total - V1 - V2 isolates interaction/excess beyond")
print("   the sum of isolated lump contributions.")
print("4. DeltaDeltaV_int compares the interaction excess of the non-Abelian")
print("   family against the Abelianized family.")
print("5. The finite-size survey probes robustness of these excess channels")
print("   in finite lattices.")
print("6. No claim is made here about continuum limits, RG invariance,")
print("   Hamiltonian spectra, or a full proof of the Yang-Mills mass gap.")
