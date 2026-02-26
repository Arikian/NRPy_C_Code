/*
 * bah_diagnostics_akv_spin.c
 *
 * Compilable AKV diagnostic scaffold with two workflows:
 *   (A) L1-reduced AKV: assemble 3x3 generalized eigenproblem (H x = λ N x) and solve.
 *   (B) Gridpoint-basis AKV: assemble dense K,B from bilinear-form/stencil callbacks, enforce mean-zero
 *       via an explicit (N-1) basis, solve generalized eigenproblem in that subspace, and postprocess.
 *
 * Geometric/equation content is intentionally excluded and must be supplied via callbacks.
 *
 * Contracts:
 *
 * 1) L1-reduced integrands:
 *    - The per-point Jm integrand MUST NOT include the global 1/(8π) factor.
 *    - The callback returns integrands (no quadrature weight applied inside callback).
 *
 * 2) Gridpoint stencil/bilinear assembly callback:
 *    - i_row is an interior DOF index in [0, N_theta*N_phi).
 *    - DOF ordering is row-major over interior points:
 *        it = i_row / N_phi, ip = i_row % N_phi,
 *      where it=0..N_theta-1 and ip=0..N_phi-1 are interior indices.
 *    - A canonical mapping to full ghosted-grid indices is:
 *        it_full = it + NG_theta, ip_full = ip + NG_phi, p = it_full*N_phi_tot + ip_full.
 *    - Neighbor indices in j_cols must use the same interior DOF indexing as i_row.
 *    - For a given i_row, (i_row, j) pairs should be unique. If duplicates appear, they are summed.
 *
 * 3) Dense storage:
 *    - Dense matrices are stored column-major: M[i + N*j].
 *
 * Build (standalone demo):
 *   gcc -O2 -std=c11 -DAKV_STANDALONE bah_diagnostics_akv_spin.c -lm -o akv_demo
 *
 * Optional OpenMP:
 *   gcc -O2 -std=c11 -DAKV_STANDALONE -fopenmp bah_diagnostics_akv_spin.c -lm -o akv_demo
 *
 * Optional LAPACKE:
 *   gcc -O2 -std=c11 -DUSE_LAPACKE -DAKV_STANDALONE bah_diagnostics_akv_spin.c -llapacke -llapack -lblas -lm -o akv_demo
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_LAPACKE
#include <lapacke.h>
#endif

#ifndef AKV_REAL
#define AKV_REAL double
#endif

// -----------------------------
// Error codes
// -----------------------------
typedef enum {
  AKV_SUCCESS = 0,
  AKV_ERR_NULLPTR = 1,
  AKV_ERR_BADPARAM = 2,
  AKV_ERR_ALLOC = 3,
  AKV_ERR_EIGEN_FAIL = 4,
  AKV_ERR_NOT_IMPLEMENTED = 5
} akv_error_t;

// -----------------------------
// Method selection
// -----------------------------
typedef enum {
  AKV_METHOD_L1_REDUCED = 0,
  AKV_METHOD_GRIDPOINT  = 1
} akv_method_t;

// -----------------------------
// Horizon grid description
// -----------------------------
typedef struct {
  int N_theta, N_phi;     // interior
  int NG_theta, NG_phi;   // ghost zones
  AKV_REAL dtheta, dphi;

  int N_theta_tot;
  int N_phi_tot;
  int N_tot;

  // Quadrature weights on the full (ghosted) grid (ghosts unused in integration).
  const AKV_REAL *w; // length N_tot

  // Optional per-point geometry pointer required by user callbacks.
  const void *geom;
} akv_horizon_grid_t;

// -----------------------------
// Diagnostics output
// -----------------------------
typedef struct {
  AKV_REAL akv_lambda[3];
  AKV_REAL akv_J[3];
  AKV_REAL akv_a[3];
  AKV_REAL akv_spin_vec[3];
  AKV_REAL akv_eig_gap_43;
  AKV_REAL akv_eig_resid[3];

  // Quality flag bitfield (0 = OK):
  //   bit0: residual tolerance exceeded for any reported mode
  //   bit1: small eigenvalue gap (gridpoint method)
  //   bit2: solver/library unavailable
  //   bit3: B not SPD / regularization retry used / near-singular handling invoked
  int akv_quality_flag;

  akv_method_t method_used;

  // L1-reduced: generalized eigenvectors (columns) in rigid-rotation coefficient basis.
  AKV_REAL akv_l1_evecs[3][3];

  // Gridpoint: store the lowest three full-space mean-zero eigenvectors z(p) (interior DOF space).
  // These are allocated by the solver; call akv_diagnostics_free() to release.
  int gp_N;                 // = N_theta*N_phi if available
  AKV_REAL *gp_z[3];        // each length gp_N, or NULL if not computed/available
} akv_diagnostics_t;

void akv_diagnostics_free(akv_diagnostics_t *d) {
  if (!d) return;
  for (int a = 0; a < 3; a++) {
    free(d->gp_z[a]);
    d->gp_z[a] = NULL;
  }
  d->gp_N = 0;
}

// -----------------------------
// Runtime parameters
// -----------------------------
typedef struct {
  akv_method_t method;

  int l1_choose_index;

  int full_num_eigs;     // >=4 recommended for gap diagnostic
  AKV_REAL eig_tol;      // e.g., 1e-10 to 1e-8
  AKV_REAL reg_eps;      // initial diagonal regularization for B_red
  AKV_REAL reg_eps_max = NAN;  // maximum regularization allowed
  int reg_max_tries;     // retry count for SPD failures
  AKV_REAL gap_ratio_thresh; // e.g., 1.5
  AKV_REAL horizon_area; // the area of the black hole horizon

  bool build_spin_vector;

  // If true and stencil callback is absent, allow debug-only O(N^3) assembly via applyK/applyB.
  bool allow_debug_assembly;
} akv_params_t;

// -----------------------------
// Callbacks (equations intentionally omitted)
// -----------------------------

// L1-reduced per-point integrands (no quadrature weight applied inside callback).
// Jm integrand must NOT include the global 1/(8π) factor.
typedef void (*akv_eval_l1_integrands_f)(
    int p,
    const akv_horizon_grid_t *grid,
    AKV_REAL Hmn[3][3],
    AKV_REAL Nmn[3][3],
    AKV_REAL Jm[3]);

// Optional mapping: given 3 coefficients, compute background-frame spin vector S_out[3].
typedef void (*akv_map_to_spinvec_f)(
    const AKV_REAL coeffs[3],
    const akv_horizon_grid_t *grid,
    AKV_REAL S_out[3]);

// Optional sign-fix evaluator for L1-reduced modes.
// Return a reference scalar that should be positive after sign convention is applied.
typedef AKV_REAL (*akv_eval_sign_l1_f)(
    const AKV_REAL coeffs[3],
    const AKV_REAL Jm_assembled[3],
    const akv_horizon_grid_t *grid);

// Gridpoint operator hooks (optional debug-only assembly and/or residual checks without dense matrices).
typedef void (*akv_apply_K_f)(const akv_horizon_grid_t *grid, const AKV_REAL *z_in, AKV_REAL *Kz_out);
typedef void (*akv_apply_B_f)(const akv_horizon_grid_t *grid, const AKV_REAL *z_in, AKV_REAL *Bz_out);

// Gridpoint-basis stencil/bilinear-form row evaluation for dense matrix assembly.
typedef int (*akv_eval_row_stencil_f)(
    int i_row,
    const akv_horizon_grid_t *grid,
    int max_entries,
    int *j_cols,
    AKV_REAL *K_vals,
    AKV_REAL *B_vals);

// Optional sign-fix evaluator for gridpoint modes, given the full-space mean-zero eigenvector z (interior DOF space).
typedef AKV_REAL (*akv_eval_sign_full_f)(
    const AKV_REAL *z_full,
    int N_full,
    const akv_horizon_grid_t *grid);

// Optional angular momentum integrand callback for gridpoint eigenmodes (loop scaffold).
// Return the per-point integrand (no weight, no 1/(8π)); caller multiplies by w[p].
typedef AKV_REAL (*akv_eval_J_integrand_full_f)(
    int p,
    const akv_horizon_grid_t *grid,
    const AKV_REAL *z_full,
    int N_full);

// -----------------------------
// Indexing helpers
// -----------------------------
static inline int akv_idx2_full(const akv_horizon_grid_t *g, int it_full, int ip_full) {
  return it_full * g->N_phi_tot + ip_full;
}

// Interior DOF index -> interior (it,ip)
static inline void akv_dof_to_it_ip(const akv_horizon_grid_t *g, int dof, int *it, int *ip) {
  (void)g;
  *it = dof / g->N_phi;
  *ip = dof - (*it) * g->N_phi;
}

// Interior (it,ip) -> interior DOF index
static inline int akv_it_ip_to_dof(const akv_horizon_grid_t *g, int it, int ip) {
  return it * g->N_phi + ip;
}

// Interior DOF -> full ghosted p
static inline int akv_dof_to_full_p(const akv_horizon_grid_t *g, int dof) {
  int it, ip;
  akv_dof_to_it_ip(g, dof, &it, &ip);
  int it_full = it + g->NG_theta;
  int ip_full = ip + g->NG_phi;
  return akv_idx2_full(g, it_full, ip_full);
}

// -----------------------------
// Small linear algebra helpers
// -----------------------------
static inline void mat3_zero(AKV_REAL A[3][3]) { memset(A, 0, 9 * sizeof(AKV_REAL)); }
static inline void vec3_zero(AKV_REAL v[3]) { memset(v, 0, 3 * sizeof(AKV_REAL)); }

static inline void mat3_add_inplace(AKV_REAL A[3][3], const AKV_REAL B[3][3], AKV_REAL alpha) {
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) A[i][j] += alpha * B[i][j];
}
static inline void vec3_add_inplace(AKV_REAL a[3], const AKV_REAL b[3], AKV_REAL alpha) {
  for (int i = 0; i < 3; i++) a[i] += alpha * b[i];
}
static inline AKV_REAL vec3_dot(const AKV_REAL a[3], const AKV_REAL b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static inline AKV_REAL vecn_dot(const AKV_REAL *a, const AKV_REAL *b, int n) {
  AKV_REAL s = 0;
  for (int i = 0; i < n; i++) s += a[i]*b[i];
  return s;
}
static inline AKV_REAL vecn_norm2(const AKV_REAL *a, int n) {
  return sqrt(vecn_dot(a, a, n));
}
static inline void vecn_scale(AKV_REAL *a, int n, AKV_REAL alpha) {
  for (int i = 0; i < n; i++) a[i] *= alpha;
}
static inline void vecn_axpy(AKV_REAL *y, const AKV_REAL *x, int n, AKV_REAL alpha) {
  for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}

static void mat3_mul_vec(const AKV_REAL A[3][3], const AKV_REAL x[3], AKV_REAL y[3]) {
  for (int i = 0; i < 3; i++) {
    y[i] = 0;
    for (int j = 0; j < 3; j++) y[i] += A[i][j] * x[j];
  }
}

static bool chol3_lower(const AKV_REAL N[3][3], AKV_REAL L[3][3]) {
  mat3_zero(L);
  AKV_REAL a00 = N[0][0];
  if (a00 <= 0) return false;
  L[0][0] = sqrt(a00);

  L[1][0] = N[1][0] / L[0][0];
  L[2][0] = N[2][0] / L[0][0];

  AKV_REAL a11 = N[1][1] - L[1][0]*L[1][0];
  if (a11 <= 0) return false;
  L[1][1] = sqrt(a11);

  L[2][1] = (N[2][1] - L[2][0]*L[1][0]) / L[1][1];

  AKV_REAL a22 = N[2][2] - L[2][0]*L[2][0] - L[2][1]*L[2][1];
  if (a22 <= 0) return false;
  L[2][2] = sqrt(a22);

  return true;
}

static void tri3_solve_lower(const AKV_REAL L[3][3], const AKV_REAL b[3], AKV_REAL y[3]) {
  y[0] = b[0] / L[0][0];
  y[1] = (b[1] - L[1][0]*y[0]) / L[1][1];
  y[2] = (b[2] - L[2][0]*y[0] - L[2][1]*y[1]) / L[2][2];
}
static void tri3_solve_upper_from_lowerT(const AKV_REAL L[3][3], const AKV_REAL y[3], AKV_REAL x[3]) {
  x[2] = y[2] / L[2][2];
  x[1] = (y[1] - L[2][1]*x[2]) / L[1][1];
  x[0] = (y[0] - L[1][0]*x[1] - L[2][0]*x[2]) / L[0][0];
}

static void generalized_to_standard3(const AKV_REAL H[3][3], const AKV_REAL L[3][3], AKV_REAL C[3][3]) {
  AKV_REAL A[3][3];

  for (int i = 0; i < 3; i++) {
    AKV_REAL h[3] = { H[0][i], H[1][i], H[2][i] };
    AKV_REAL a[3];
    tri3_solve_lower(L, h, a); // a = L^-1 h
    A[0][i] = a[0]; A[1][i] = a[1]; A[2][i] = a[2]; // A = L^-1 H
  }

  for (int i = 0; i < 3; i++) {
    AKV_REAL c[3];
    tri3_solve_lower(L, A[i], c); // c = L^-1 a^T
                                  //   = L^-1 h L^-T (H is symmetric)
    C[0][i] = c[0]; C[1][i] = c[1]; C[2][i] = c[2];
  }

  // Fixes asymmetry that may be caused by floating point precision.
  for (int i = 0; i < 3; i++) for (int j = i+1; j < 3; j++) {
    AKV_REAL s = (C[i][j] + C[j][i])/2;
    C[i][j] = s; C[j][i] = s;
  }
}

static void jacobi_eig_sym3(const AKV_REAL A_in[3][3], AKV_REAL eval[3], AKV_REAL V[3][3]) {
  AKV_REAL A[3][3];
  memcpy(A, A_in, 9*sizeof(AKV_REAL));

  mat3_zero(V);
  V[0][0]=1; V[1][1]=1; V[2][2]=1;

  const int max_sweeps = 50;
  const AKV_REAL tol = 1e-14;

  for (int sweep = 0; sweep < max_sweeps; sweep++) {
    int p=0,q=1;
    AKV_REAL maxv = fabs(A[0][1]);
    if (fabs(A[0][2]) > maxv) { maxv = fabs(A[0][2]); p=0; q=2; }
    if (fabs(A[1][2]) > maxv) { maxv = fabs(A[1][2]); p=1; q=2; }
    if (maxv < tol) break;

    AKV_REAL app = A[p][p];
    AKV_REAL aqq = A[q][q];
    AKV_REAL apq = A[p][q];

    AKV_REAL tau = (aqq - app) / (2.0 * apq);
    AKV_REAL t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
    AKV_REAL c = 1.0 / sqrt(1.0 + t*t);
    AKV_REAL s = t * c;

    A[p][p] = app - t*apq;
    A[q][q] = aqq + t*apq;
    A[p][q] = 0.0; A[q][p] = 0.0;

    for (int r = 0; r < 3; r++) {
      if (r == p || r == q) continue;
      AKV_REAL arp = A[r][p];
      AKV_REAL arq = A[r][q];
      A[r][p] = c*arp - s*arq; A[p][r] = A[r][p];
      A[r][q] = c*arq + s*arp; A[q][r] = A[r][q];
    }

    for (int r = 0; r < 3; r++) {
      AKV_REAL vrp = V[r][p];
      AKV_REAL vrq = V[r][q];
      V[r][p] = c*vrp - s*vrq;
      V[r][q] = c*vrq + s*vrp;
    }
  }

  eval[0] = A[0][0];
  eval[1] = A[1][1];
  eval[2] = A[2][2];
}

static void sort_eigs3(AKV_REAL eval[3], AKV_REAL V[3][3]) {
  for (int i = 0; i < 3; i++) {
    int k = i;
    for (int j = i+1; j < 3; j++) if (eval[j] < eval[k]) k = j;
    if (k != i) {
      AKV_REAL te = eval[i]; eval[i] = eval[k]; eval[k] = te;
      for (int r = 0; r < 3; r++) {
        AKV_REAL tv = V[r][i]; V[r][i] = V[r][k]; V[r][k] = tv;
      }
    }
  }
}

static void normalize_gen_evec3(const AKV_REAL N[3][3], AKV_REAL x[3]) {
  AKV_REAL Nx[3];
  mat3_mul_vec(N, x, Nx);
  AKV_REAL nrm2 = vec3_dot(x, Nx);
  if (nrm2 > 0) {
    AKV_REAL inv = 1.0 / sqrt(nrm2);
    x[0] *= inv; x[1] *= inv; x[2] *= inv;
  }
}

// -----------------------------
// Gridpoint mean-zero basis utilities with cached map and buffers
// -----------------------------
typedef struct {
  int N_full;      // N_theta*N_phi
  int N_red;       // N_full - 1
  int ref;         // reference DOF index removed
  AKV_REAL *w_dof; // length N_full, per-DOF weights
  AKV_REAL wref;   // w_dof[ref]
  int *map;        // length N_red, reduced index -> full index (skip ref)

  // Scratch buffers (reused)
  AKV_REAL *tmpN;  // length max(N_full,N_red)
  AKV_REAL *tmpK;  // length max(N_full,N_red)
  AKV_REAL *tmpR;  // length max(N_full,N_red)
} akv_mean0_basis_t;

static void akv_mean0_basis_free(akv_mean0_basis_t *b) {
  if (!b) return;
  free(b->w_dof);
  free(b->map);
  free(b->tmpN);
  free(b->tmpK);
  free(b->tmpR);
  memset(b, 0, sizeof(*b));
}

static akv_error_t akv_mean0_basis_build(const akv_horizon_grid_t *grid, akv_mean0_basis_t *b) {
  if (!grid || !b || !grid->w) return AKV_ERR_NULLPTR;
  const int N_full = grid->N_theta * grid->N_phi;
  if (N_full <= 1) return AKV_ERR_BADPARAM;

  memset(b, 0, sizeof(*b));
  b->N_full = N_full;
  b->N_red = N_full - 1;
  b->ref = -1;

  b->w_dof = (AKV_REAL*)calloc((size_t)N_full, sizeof(AKV_REAL));
  b->map   = (int*)malloc((size_t)b->N_red * sizeof(int));
  if (!b->w_dof || !b->map) { akv_mean0_basis_free(b); return AKV_ERR_ALLOC; }

  for (int i = 0; i < N_full; i++) {
    int p = akv_dof_to_full_p(grid, i);
    b->w_dof[i] = grid->w[p];
  }

  for (int i = 0; i < N_full; i++) {
    if (fabs(b->w_dof[i]) > 0) { b->ref = i; b->wref = b->w_dof[i]; break; }
  }
  if (b->ref < 0 || b->wref == 0.0) { akv_mean0_basis_free(b); return AKV_ERR_BADPARAM; }

  for (int i = 0, k = 0; i < N_full; i++) if (i != b->ref) b->map[k++] = i;

  int scratch_len = (b->N_full > b->N_red) ? b->N_full : b->N_red;
  b->tmpN = (AKV_REAL*)malloc((size_t)scratch_len * sizeof(AKV_REAL));
  b->tmpK = (AKV_REAL*)malloc((size_t)scratch_len * sizeof(AKV_REAL));
  b->tmpR = (AKV_REAL*)malloc((size_t)scratch_len * sizeof(AKV_REAL));
  if (!b->tmpN || !b->tmpK || !b->tmpR) { akv_mean0_basis_free(b); return AKV_ERR_ALLOC; }

  return AKV_SUCCESS;
}

// Transform full-space dense matrix M_full (N_full x N_full) into reduced mean-zero matrix (N_red x N_red):
// M_red = B^T M_full B, where basis columns are b_i = e_i - (w_i/w_ref) e_ref for i != ref.
static void akv_mean0_transform_dense(
    const akv_mean0_basis_t *b,
    const AKV_REAL *M_full, int ld_full,
    AKV_REAL *M_red, int ld_red) {

  const int N = b->N_full;
  const int Nr = b->N_red;
  const int r = b->ref;

  for (int a = 0; a < Nr; a++) {
    const int ia = b->map[a];
    const AKV_REAL alpha_a = -b->w_dof[ia] / b->wref;

    for (int c = 0; c < Nr; c++) {
      const int ic = b->map[c];
      const AKV_REAL alpha_c = -b->w_dof[ic] / b->wref;

      const AKV_REAL M_ia_ic = M_full[ia + (size_t)ld_full * ic];
      const AKV_REAL M_ia_r  = M_full[ia + (size_t)ld_full * r];
      const AKV_REAL M_r_ic  = M_full[r  + (size_t)ld_full * ic];
      const AKV_REAL M_r_r   = M_full[r  + (size_t)ld_full * r];

      (void)N;
      M_red[a + (size_t)ld_red * c] =
          M_ia_ic + alpha_c * M_ia_r + alpha_a * M_r_ic + (alpha_a * alpha_c) * M_r_r;
    }
  }
}

// Lift reduced vector x_red (Nr) to full mean-zero vector x_full (N_full) using the cached basis.
static void akv_mean0_lift_vector(
    const akv_mean0_basis_t *b,
    const AKV_REAL *x_red,
    AKV_REAL *x_full) {

  const int N = b->N_full;
  const int Nr = b->N_red;
  const int r = b->ref;

  memset(x_full, 0, (size_t)N * sizeof(AKV_REAL));

  AKV_REAL xref = 0.0;
  for (int a = 0; a < Nr; a++) {
    const int ia = b->map[a];
    x_full[ia] = x_red[a];
    xref += (-b->w_dof[ia] / b->wref) * x_red[a];
  }
  x_full[r] = xref;
}

// -----------------------------
// Dense helpers (column-major)
// -----------------------------
static void akv_symmetrize_dense(AKV_REAL *M, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      AKV_REAL s = 0.5*(M[i + (size_t)N*j] + M[j + (size_t)N*i]);
      M[i + (size_t)N*j] = s;
      M[j + (size_t)N*i] = s;
    }
  }
}

static void akv_dense_add_diag(AKV_REAL *M, int N, AKV_REAL eps) {
  for (int i = 0; i < N; i++) M[i + (size_t)N*i] += eps;
}

static void akv_dense_matvec(const AKV_REAL *M, int N, const AKV_REAL *x, AKV_REAL *y) {
  for (int i = 0; i < N; i++) {
    AKV_REAL s = 0.0;
    const AKV_REAL *col = &M[i]; // M[i + N*j]
    for (int j = 0; j < N; j++) s += col[(size_t)N*j] * x[j];
    y[i] = s;
  }
}

// Residual ratio for dense matrices using caller-provided scratch buffers of length N.
static AKV_REAL akv_residual_ratio_dense_inplace(
    const AKV_REAL *K, const AKV_REAL *B, int N,
    const AKV_REAL *z, AKV_REAL lambda,
    AKV_REAL *Kz, AKV_REAL *Bz, AKV_REAL *r) {

  akv_dense_matvec(K, N, z, Kz);
  akv_dense_matvec(B, N, z, Bz);
  for (int i = 0; i < N; i++) r[i] = Kz[i] - lambda * Bz[i];

  AKV_REAL nr = vecn_norm2(r, N);
  AKV_REAL nK = vecn_norm2(Kz, N);
  AKV_REAL nB = vecn_norm2(Bz, N);
  AKV_REAL denom = nK + nB;
  return (denom > 0) ? (nr / denom) : 0.0;
}

/*
// Residual ratio using operator application (no dense matrices), with caller-provided scratch buffers of length N.
static AKV_REAL akv_residual_ratio_apply_inplace(
    const akv_horizon_grid_t *grid,
    akv_apply_K_f applyK,
    akv_apply_B_f applyB,
    int N,
    const AKV_REAL *z,
    AKV_REAL lambda,
    AKV_REAL *Kz, AKV_REAL *Bz, AKV_REAL *r) {

  applyK(grid, z, Kz);
  applyB(grid, z, Bz);
  for (int i = 0; i < N; i++) r[i] = Kz[i] - lambda * Bz[i];

  AKV_REAL nr = vecn_norm2(r, N);
  AKV_REAL nK = vecn_norm2(Kz, N);
  AKV_REAL nB = vecn_norm2(Bz, N);
  AKV_REAL denom = nK + nB;
  return (denom > 0) ? (nr / denom) : 0.0;
}
*/

// B-inner-product Gram–Schmidt on k vectors (columns) in V (N x k), using dense B (N x N).
// Reuses tmp buffer of length N.
static void akv_b_gram_schmidt_inplace(AKV_REAL *V, int N, int k, const AKV_REAL *B, AKV_REAL *tmp) {
  for (int a = 0; a < k; a++) {
    AKV_REAL *va = &V[(size_t)N * a];

    for (int b = 0; b < a; b++) {
      AKV_REAL *vb = &V[(size_t)N * b];

      akv_dense_matvec(B, N, vb, tmp);
      AKV_REAL proj = vecn_dot(va, tmp, N); // <va,vb>_B
      vecn_axpy(va, vb, N, -proj);
    }

    akv_dense_matvec(B, N, va, tmp);
    AKV_REAL n2 = vecn_dot(va, tmp, N);
    if (n2 > 0) vecn_scale(va, N, 1.0 / sqrt(n2));
  }
}

// -----------------------------
// L1-reduced assembly + solve + postprocess
// -----------------------------
static AKV_REAL default_sign_ref_l1(const AKV_REAL coeffs[3], const AKV_REAL Jm_assembled[3]) {
  return coeffs[0]*Jm_assembled[0] + coeffs[1]*Jm_assembled[1] + coeffs[2]*Jm_assembled[2];
}

static akv_error_t akv_solve_l1_reduced(
    const akv_horizon_grid_t *grid,
    const akv_params_t *pars,
    akv_eval_l1_integrands_f eval_l1,
    akv_eval_sign_l1_f eval_sign_l1,
    akv_map_to_spinvec_f map_to_spinvec,
    akv_diagnostics_t *out) {

  if (!grid || !pars || !eval_l1 || !out) return AKV_ERR_NULLPTR;
  if (!grid->w) return AKV_ERR_NULLPTR;

  AKV_REAL H[3][3], N[3][3], Jm[3];
  mat3_zero(H); mat3_zero(N); vec3_zero(Jm);

#ifdef _OPENMP
  AKV_REAL H_acc[3][3], N_acc[3][3], J_acc[3];
  mat3_zero(H_acc); mat3_zero(N_acc); vec3_zero(J_acc);
#pragma omp parallel
  {
    AKV_REAL Hloc[3][3], Nloc[3][3], Jloc[3];
    mat3_zero(Hloc); mat3_zero(Nloc); vec3_zero(Jloc);

#pragma omp for collapse(2) nowait
    for (int it = grid->NG_theta; it < grid->NG_theta + grid->N_theta; it++) {
      for (int ip = grid->NG_phi; ip < grid->NG_phi + grid->N_phi; ip++) {
        const int p = akv_idx2_full(grid, it, ip);
        const AKV_REAL wp = grid->w[p];

        AKV_REAL Hij[3][3], Nij[3][3], Jpi[3];
        eval_l1(p, grid, Hij, Nij, Jpi);

        mat3_add_inplace(Hloc, Hij, wp);
        mat3_add_inplace(Nloc, Nij, wp);
        vec3_add_inplace(Jloc, Jpi, wp);
      }
    }

#pragma omp critical
    {
      mat3_add_inplace(H_acc, Hloc, 1.0);
      mat3_add_inplace(N_acc, Nloc, 1.0);
      vec3_add_inplace(J_acc, Jloc, 1.0);
    }
  }
  memcpy(H, H_acc, 9*sizeof(AKV_REAL));
  memcpy(N, N_acc, 9*sizeof(AKV_REAL));
  memcpy(Jm, J_acc, 3*sizeof(AKV_REAL));
#else
  for (int it = grid->NG_theta; it < grid->NG_theta + grid->N_theta; it++) {
    for (int ip = grid->NG_phi; ip < grid->NG_phi + grid->N_phi; ip++) {
      const int p = akv_idx2_full(grid, it, ip);
      const AKV_REAL wp = grid->w[p];

      AKV_REAL Hij[3][3], Nij[3][3], Jpi[3];
      eval_l1(p, grid, Hij, Nij, Jpi);

      mat3_add_inplace(H, Hij, wp);
      mat3_add_inplace(N, Nij, wp);
      vec3_add_inplace(Jm, Jpi, wp);
    }
  }
#endif

  for (int i = 0; i < 3; i++) for (int j = i+1; j < 3; j++) {
    H[i][j] = H[j][i] = 0.5*(H[i][j] + H[j][i]);
    N[i][j] = N[j][i] = 0.5*(N[i][j] + N[j][i]);
  }

  if (pars->reg_eps > 0) {
    N[0][0] += pars->reg_eps;
    N[1][1] += pars->reg_eps;
    N[2][2] += pars->reg_eps;
  }

  AKV_REAL L[3][3];
  if (!chol3_lower(N, L)) {
    out->akv_quality_flag |= (1<<3);
    return AKV_ERR_EIGEN_FAIL;
  }

  AKV_REAL C[3][3];
  generalized_to_standard3(H, L, C);

  AKV_REAL eval[3], V[3][3];
  jacobi_eig_sym3(C, eval, V);
  sort_eigs3(eval, V);

  out->akv_lambda[0] = eval[0];
  out->akv_lambda[1] = eval[1];
  out->akv_lambda[2] = eval[2];

  for (int k = 0; k < 3; k++) {
    AKV_REAL u[3] = { V[0][k], V[1][k], V[2][k] };
    AKV_REAL x[3];
    tri3_solve_upper_from_lowerT(L, u, x);
    normalize_gen_evec3(N, x);

    AKV_REAL sref = 0.0;
    if (eval_sign_l1) sref = eval_sign_l1(x, Jm, grid);
    else sref = default_sign_ref_l1(x, Jm);
    if (sref < 0) { x[0] = -x[0]; x[1] = -x[1]; x[2] = -x[2]; }

    out->akv_l1_evecs[0][k] = x[0];
    out->akv_l1_evecs[1][k] = x[1];
    out->akv_l1_evecs[2][k] = x[2];

    AKV_REAL Hx[3], Nx[3], r[3];
    mat3_mul_vec(H, x, Hx);
    mat3_mul_vec(N, x, Nx);
    r[0] = Hx[0] - eval[k]*Nx[0];
    r[1] = Hx[1] - eval[k]*Nx[1];
    r[2] = Hx[2] - eval[k]*Nx[2];

    AKV_REAL nr = sqrt(vec3_dot(r, r));
    AKV_REAL nHx = sqrt(vec3_dot(Hx, Hx));
    AKV_REAL nNx = sqrt(vec3_dot(Nx, Nx));
    AKV_REAL denom = nHx + nNx;
    out->akv_eig_resid[k] = (denom > 0) ? (nr / denom) : 0.0;
    if (pars->eig_tol > 0 && out->akv_eig_resid[k] > pars->eig_tol) out->akv_quality_flag |= (1<<0);
  }

  out->akv_J[0] = Jm[0]/(8*M_PI);
  out->akv_J[1] = Jm[1]/(8*M_PI);
  out->akv_J[2] = Jm[2]/(8*M_PI);


  static const AKV_REAL A = pars->horizon_area;
  const AKV_REAL DLSF = 2*A/(A*A+Jm[0]+Jm[1]+Jm[2]);
  out->akv_a[0] = Jm[0]*DSLF;
  out->akv_a[1] = Jm[1]*DSLF;
  out->akv_a[2] = Jm[2]*DSLF;

  if (pars->build_spin_vector && map_to_spinvec) {
    int k = pars->l1_choose_index;
    if (k < 0) k = 0;
    if (k > 2) k = 2;
    AKV_REAL coeffs[3] = { out->akv_l1_evecs[0][k], out->akv_l1_evecs[1][k], out->akv_l1_evecs[2][k] };
    map_to_spinvec(coeffs, grid, out->akv_spin_vec);
  } else {
    out->akv_spin_vec[0] = 0.0;
    out->akv_spin_vec[1] = 0.0;
    out->akv_spin_vec[2] = 0.0;
  }

  out->akv_eig_gap_43 = 0.0;
  out->method_used = AKV_METHOD_L1_REDUCED;

  return AKV_SUCCESS;
}

// -----------------------------
// Gridpoint dense assembly
// -----------------------------

typedef struct {
  int j;
  AKV_REAL Kv;
  AKV_REAL Bv;
} akv_triplet_t;

static int cmp_triplet_j(const void *a, const void *b) {
  const akv_triplet_t *x = (const akv_triplet_t*)a;
  const akv_triplet_t *y = (const akv_triplet_t*)b;
  return (x->j < y->j) ? -1 : (x->j > y->j);
}

// Assemble dense matrices from stencil/bilinear-form callback.
// For each row i, the callback provides (i,j,K,B) entries; duplicates in j are summed.
static akv_error_t akv_full_assemble_dense_from_stencil(
    const akv_horizon_grid_t *grid,
    akv_eval_row_stencil_f eval_row,
    AKV_REAL *Kmat, AKV_REAL *Bmat, int N_full) {

  if (!grid || !eval_row || !Kmat || !Bmat) return AKV_ERR_NULLPTR;

  const int max_entries = 512;
  int *j_cols = (int*)malloc((size_t)max_entries * sizeof(int));
  AKV_REAL *K_vals = (AKV_REAL*)malloc((size_t)max_entries * sizeof(AKV_REAL));
  AKV_REAL *B_vals = (AKV_REAL*)malloc((size_t)max_entries * sizeof(AKV_REAL));
  akv_triplet_t *t = (akv_triplet_t*)malloc((size_t)max_entries * sizeof(akv_triplet_t));
  if (!j_cols || !K_vals || !B_vals || !t) {
    free(j_cols); free(K_vals); free(B_vals); free(t);
    return AKV_ERR_ALLOC;
  }

  for (int i = 0; i < N_full; i++) {
    int nnz = eval_row(i, grid, max_entries, j_cols, K_vals, B_vals);
    if (nnz < 0) { free(j_cols); free(K_vals); free(B_vals); free(t); return AKV_ERR_BADPARAM; }
    if (nnz > max_entries) nnz = max_entries;

    int m = 0;
    for (int e = 0; e < nnz; e++) {
      int j = j_cols[e];
      if (j < 0 || j >= N_full) continue;
      t[m].j = j;
      t[m].Kv = K_vals[e];
      t[m].Bv = B_vals[e];
      m++;
    }

    if (m == 0) continue;

    qsort(t, (size_t)m, sizeof(akv_triplet_t), cmp_triplet_j);

    int curj = t[0].j;
    AKV_REAL sumK = t[0].Kv;
    AKV_REAL sumB = t[0].Bv;

    for (int e = 1; e < m; e++) {
      if (t[e].j == curj) {
        sumK += t[e].Kv;
        sumB += t[e].Bv;
      } else {
        Kmat[i + (size_t)N_full * curj] = sumK;
        Bmat[i + (size_t)N_full * curj] = sumB;
        curj = t[e].j;
        sumK = t[e].Kv;
        sumB = t[e].Bv;
      }
    }
    Kmat[i + (size_t)N_full * curj] = sumK;
    Bmat[i + (size_t)N_full * curj] = sumB;
  }

  free(j_cols); free(K_vals); free(B_vals); free(t);

  akv_symmetrize_dense(Kmat, N_full);
  akv_symmetrize_dense(Bmat, N_full);
  return AKV_SUCCESS;
}

// Debug-only dense assembly by applying operator to basis vectors (O(N^3) operator work).
static akv_error_t akv_full_assemble_dense_debug_apply(
    const akv_horizon_grid_t *grid,
    akv_apply_K_f applyK,
    akv_apply_B_f applyB,
    AKV_REAL *Kmat, AKV_REAL *Bmat, int N_full) {

  if (!grid || !applyK || !applyB || !Kmat || !Bmat) return AKV_ERR_NULLPTR;

  AKV_REAL *e = (AKV_REAL*)calloc((size_t)N_full, sizeof(AKV_REAL));
  AKV_REAL *Kz = (AKV_REAL*)calloc((size_t)N_full, sizeof(AKV_REAL));
  AKV_REAL *Bz = (AKV_REAL*)calloc((size_t)N_full, sizeof(AKV_REAL));
  if (!e || !Kz || !Bz) {
    free(e); free(Kz); free(Bz);
    return AKV_ERR_ALLOC;
  }

  for (int n = 0; n < N_full; n++) {
    memset(e, 0, (size_t)N_full * sizeof(AKV_REAL));
    e[n] = 1.0;
    applyK(grid, e, Kz);
    applyB(grid, e, Bz);
    for (int m = 0; m < N_full; m++) {
      Kmat[m + (size_t)N_full * n] = Kz[m];
      Bmat[m + (size_t)N_full * n] = Bz[m];
    }
  }

  akv_symmetrize_dense(Kmat, N_full);
  akv_symmetrize_dense(Bmat, N_full);

  free(e); free(Kz); free(Bz);
  return AKV_SUCCESS;
}

// -----------------------------
// Gridpoint dense generalized eigen solve with explicit SPD handling
// -----------------------------

#ifdef USE_LAPACKE
static bool akv_try_cholesky_spd(const AKV_REAL *B, int N) {
  AKV_REAL *tmp = (AKV_REAL*)malloc((size_t)N*(size_t)N*sizeof(AKV_REAL));
  if (!tmp) return false;
  memcpy(tmp, B, (size_t)N*(size_t)N*sizeof(AKV_REAL));
  int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', N, tmp, N);
  free(tmp);
  return (info == 0);
}
#endif

static akv_error_t akv_full_solve_dense_generalized_reduced_spd_retry(
    AKV_REAL *Kred, AKV_REAL *Bred, int Nr, int want_neigs,
    const akv_params_t *pars,
    AKV_REAL *eval_out, AKV_REAL *evec_out,
    int *quality_flag_io) {

  if (!Kred || !Bred || !eval_out || !evec_out || !pars || !quality_flag_io) return AKV_ERR_NULLPTR;
  if (Nr <= 0 || want_neigs <= 0 || want_neigs > Nr) return AKV_ERR_BADPARAM;

#ifndef USE_LAPACKE
  (void)Kred; (void)Bred; (void)Nr; (void)want_neigs; (void)pars; (void)eval_out; (void)evec_out;
  *quality_flag_io |= (1<<2);
  return AKV_ERR_NOT_IMPLEMENTED;
#else
  const int max_tries = (pars->reg_max_tries > 0) ? pars->reg_max_tries : 1;
  AKV_REAL reg = (pars->reg_eps > 0) ? pars->reg_eps : 0.0;
  const AKV_REAL reg_max = (pars->reg_eps_max > reg) ? pars->reg_eps_max : reg;

  bool max_eps = false;
  for (int attempt = 0; attempt < max_tries; attempt++) {
    AKV_REAL *A = (AKV_REAL*)malloc((size_t)Nr*(size_t)Nr*sizeof(AKV_REAL));
    AKV_REAL *B = (AKV_REAL*)malloc((size_t)Nr*(size_t)Nr*sizeof(AKV_REAL));
    AKV_REAL *w = (AKV_REAL*)malloc((size_t)Nr*sizeof(AKV_REAL));
    if (!A || !B || !w) { free(A); free(B); free(w); return AKV_ERR_ALLOC; }

    memcpy(A, Kred, (size_t)Nr*(size_t)Nr*sizeof(AKV_REAL));
    memcpy(B, Bred, (size_t)Nr*(size_t)Nr*sizeof(AKV_REAL));

    if (reg > 0) {
      akv_dense_add_diag(B, Nr, reg);
      *quality_flag_io |= (1<<3);
    }

    // Explicit SPD probe (Cholesky). If it fails, increase reg and retry.
    if (!akv_try_cholesky_spd(B, Nr)) {
      free(A); free(B); free(w);
      *quality_flag_io |= (1<<3);
      if (attempt == max_tries - 1 || max_eps) return AKV_ERR_EIGEN_FAIL;
      if (reg == 0.0) reg = 1e-14;
      else reg *= 10.0;
      if (reg_max > 0 && reg > reg_max) { reg = reg_max; max_eps = true; }
      continue;
    }
    
    // Solve A x = λ B x. LAPACKE_dsygvd overwrites A with eigenvectors.
    int info = LAPACKE_dsygvd(LAPACK_COL_MAJOR, 1, 'V', 'U', Nr, A, Nr, B, Nr, w);
    if (info != 0) {
      free(A); free(B); free(w);
      *quality_flag_io |= (1<<3);
      if (attempt == max_tries - 1 || max_eps) return AKV_ERR_EIGEN_FAIL;
      if (reg == 0.0) reg = 1e-14;
      else reg *= 10.0;
      if (reg_max > 0 && reg > reg_max) { reg = reg_max; max_eps = true; }
      continue;
    }

    for (int i = 0; i < want_neigs; i++) {
      eval_out[i] = w[i];
      for (int r = 0; r < Nr; r++) evec_out[r + (size_t)Nr*i] = A[r + (size_t)Nr*i];
    }

    free(A); free(B); free(w);
    return AKV_SUCCESS;
  }

  return AKV_ERR_EIGEN_FAIL;
#endif
}

// -----------------------------
// Gridpoint solve + postprocess + eigenvector storage + J loop scaffold
// -----------------------------
static akv_error_t akv_solve_gridpoint_basis(
    const akv_horizon_grid_t *grid,
    const akv_params_t *pars,
    akv_eval_row_stencil_f eval_row_stencil,
    akv_apply_K_f applyK_debug,
    akv_apply_B_f applyB_debug,
    akv_eval_sign_full_f eval_sign_full,
    akv_eval_J_integrand_full_f eval_J_full,
    akv_diagnostics_t *out) {

  if (!grid || !pars || !out) return AKV_ERR_NULLPTR;
  if (!grid->w) return AKV_ERR_NULLPTR;

  const int N_full = grid->N_theta * grid->N_phi;
  if (N_full <= 1) return AKV_ERR_BADPARAM;

  akv_mean0_basis_t b;
  akv_error_t err = akv_mean0_basis_build(grid, &b);
  if (err != AKV_SUCCESS) return err;

  const int Nr = b.N_red;

  // Dense memory warning: two N_full^2 matrices may be large at typical resolutions.
  AKV_REAL *Kfull = (AKV_REAL*)calloc((size_t)N_full*(size_t)N_full, sizeof(AKV_REAL));
  AKV_REAL *Bfull = (AKV_REAL*)calloc((size_t)N_full*(size_t)N_full, sizeof(AKV_REAL));
  if (!Kfull || !Bfull) {
    free(Kfull); free(Bfull);
    akv_mean0_basis_free(&b);
    return AKV_ERR_ALLOC;
  }

  if (eval_row_stencil) {
    err = akv_full_assemble_dense_from_stencil(grid, eval_row_stencil, Kfull, Bfull, N_full);
    if (err != AKV_SUCCESS) {
      free(Kfull); free(Bfull);
      akv_mean0_basis_free(&b);
      return err;
    }
  } else {
    if (!pars->allow_debug_assembly || !applyK_debug || !applyB_debug) {
      out->akv_quality_flag |= (1<<2);
      free(Kfull); free(Bfull);
      akv_mean0_basis_free(&b);
      return AKV_ERR_BADPARAM;
    }
    err = akv_full_assemble_dense_debug_apply(grid, applyK_debug, applyB_debug, Kfull, Bfull, N_full);
    if (err != AKV_SUCCESS) {
      free(Kfull); free(Bfull);
      akv_mean0_basis_free(&b);
      return err;
    }
  }

  AKV_REAL *Kred = (AKV_REAL*)calloc((size_t)Nr*(size_t)Nr, sizeof(AKV_REAL));
  AKV_REAL *Bred = (AKV_REAL*)calloc((size_t)Nr*(size_t)Nr, sizeof(AKV_REAL));
  if (!Kred || !Bred) {
    free(Kfull); free(Bfull); free(Kred); free(Bred);
    akv_mean0_basis_free(&b);
    return AKV_ERR_ALLOC;
  }

  akv_mean0_transform_dense(&b, Kfull, N_full, Kred, Nr);
  akv_mean0_transform_dense(&b, Bfull, N_full, Bred, Nr);
  akv_symmetrize_dense(Kred, Nr);
  akv_symmetrize_dense(Bred, Nr);

  int want = pars->full_num_eigs;
  if (want < 4) want = 4;
  if (want > Nr) want = Nr;

  AKV_REAL *eval = (AKV_REAL*)malloc((size_t)want * sizeof(AKV_REAL));
  AKV_REAL *evec = (AKV_REAL*)malloc((size_t)Nr*(size_t)want * sizeof(AKV_REAL));
  if (!eval || !evec) {
    free(Kfull); free(Bfull); free(Kred); free(Bred); free(eval); free(evec);
    akv_mean0_basis_free(&b);
    return AKV_ERR_ALLOC;
  }

  err = akv_full_solve_dense_generalized_reduced_spd_retry(Kred, Bred, Nr, want, pars, eval, evec, &out->akv_quality_flag);
  if (err != AKV_SUCCESS) {
    if (err == AKV_ERR_NOT_IMPLEMENTED) out->akv_quality_flag |= (1<<2);
    free(Kfull); free(Bfull); free(Kred); free(Bred); free(eval); free(evec);
    akv_mean0_basis_free(&b);
    return err;
  }

  // B-orthonormalize first three modes using Bred.
  const int kmodes = (want >= 3) ? 3 : want;
  akv_b_gram_schmidt_inplace(evec, Nr, kmodes, Bred, b.tmpN);

  // Gap diagnostic R_{3,4} = λ4/λ3 in mean-zero subspace.
  if (want >= 4 && eval[2] != 0.0) out->akv_eig_gap_43 = eval[3] / eval[2];
  else out->akv_eig_gap_43 = 0.0;

  if (pars->gap_ratio_thresh > 0 && want >= 4 && out->akv_eig_gap_43 > 0 &&
      out->akv_eig_gap_43 < pars->gap_ratio_thresh) {
    out->akv_quality_flag |= (1<<1);
  }

  out->akv_lambda[0] = (want > 0 ? eval[0] : 0.0);
  out->akv_lambda[1] = (want > 1 ? eval[1] : 0.0);
  out->akv_lambda[2] = (want > 2 ? eval[2] : 0.0);

  // Residual ratios for first three modes using dense reduced matrices and reused buffers.
  for (int a = 0; a < 3; a++) {
    if (a >= want) { out->akv_eig_resid[a] = 0.0; continue; }
    const AKV_REAL *za = &evec[(size_t)Nr * a];
    out->akv_eig_resid[a] = akv_residual_ratio_dense_inplace(Kred, Bred, Nr, za, eval[a], b.tmpK, b.tmpN, b.tmpR);
    if (pars->eig_tol > 0 && out->akv_eig_resid[a] > pars->eig_tol) out->akv_quality_flag |= (1<<0);
  }

  // Store full-space eigenvectors (interior DOF space) for downstream postprocessing.
  out->gp_N = N_full;
  for (int a = 0; a < 3; a++) {
    out->gp_z[a] = NULL;
    if (a >= kmodes) continue;
    out->gp_z[a] = (AKV_REAL*)malloc((size_t)N_full * sizeof(AKV_REAL));
    if (!out->gp_z[a]) {
      for (int k = 0; k < a; k++) { free(out->gp_z[k]); out->gp_z[k] = NULL; }
      out->gp_N = 0;
      free(Kfull); free(Bfull); free(Kred); free(Bred); free(eval); free(evec);
      akv_mean0_basis_free(&b);
      return AKV_ERR_ALLOC;
    }
    akv_mean0_lift_vector(&b, &evec[(size_t)Nr * a], out->gp_z[a]);
  }

  // Sign-fix in full space (if provided). Flip stored gp_z and the reduced eigenvector consistently.
  for (int a = 0; a < kmodes; a++) {
    if (!out->gp_z[a]) continue;
    if (!eval_sign_full) continue;

    AKV_REAL sref = eval_sign_full(out->gp_z[a], N_full, grid);
    if (sref < 0) {
      // Flip reduced eigenvector
      for (int i = 0; i < Nr; i++) evec[i + (size_t)Nr * a] = -evec[i + (size_t)Nr * a];
      // Re-lift
      akv_mean0_lift_vector(&b, &evec[(size_t)Nr * a], out->gp_z[a]);
    }
  }

  // Loop scaffold for angular momentum integrals J[a] using per-mode z_full.
  // J integrand callback must not include 1/(8π). This function applies weights and sums.
  for (int a = 0; a < 3; a++) out->akv_J[a] = 0.0;
  if (eval_J_full) {
    for (int a = 0; a < kmodes; a++) {
      const AKV_REAL *z = out->gp_z[a];
      if (!z) continue;

      AKV_REAL sum = 0.0;
      for (int dof = 0; dof < N_full; dof++) {
        int p = akv_dof_to_full_p(grid, dof);
        AKV_REAL wp = grid->w[p];
        AKV_REAL integrand = eval_J_full(p, grid, z, N_full);
        sum += wp * integrand;
      }
      out->akv_J[a] = sum/(8*M_PI);
    }
  }

  // Dimensionless spin components normalized by M_H^2.
  static const AKV_REAL A = pars->horizon_area;
  const AKV_REAL DLSF = 2*A/(A*A+Jm[0]+Jm[1]+Jm[2]);
  out->akv_a[0] = Jm[0]*DSLF;
  out->akv_a[1] = Jm[1]*DSLF;
  out->akv_a[2] = Jm[2]*DSLF;

  out->akv_spin_vec[0] = 0.0;
  out->akv_spin_vec[1] = 0.0;
  out->akv_spin_vec[2] = 0.0;

  out->method_used = AKV_METHOD_GRIDPOINT;

  free(Kfull); free(Bfull);
  free(Kred); free(Bred);
  free(eval); free(evec);
  akv_mean0_basis_free(&b);
  return AKV_SUCCESS;
}

// -----------------------------
// Public entry point
// -----------------------------
akv_error_t akv_compute(
    const akv_horizon_grid_t *grid,
    const akv_params_t *pars,
    // L1-reduced callbacks
    akv_eval_l1_integrands_f eval_l1,
    akv_eval_sign_l1_f eval_sign_l1,
    akv_map_to_spinvec_f map_to_spinvec,
    // Gridpoint callbacks
    akv_eval_row_stencil_f eval_row_stencil,
    akv_apply_K_f applyK_debug,
    akv_apply_B_f applyB_debug,
    akv_eval_sign_full_f eval_sign_full,
    akv_eval_J_integrand_full_f eval_J_full,
    akv_diagnostics_t *out) {

  if (!grid || !pars || !out) return AKV_ERR_NULLPTR;
  memset(out, 0, sizeof(*out));

  switch (pars->method) {
    case AKV_METHOD_L1_REDUCED:
      return akv_solve_l1_reduced(grid, pars, eval_l1, eval_sign_l1, map_to_spinvec, out);
    case AKV_METHOD_GRIDPOINT:
      return akv_solve_gridpoint_basis(grid, pars, eval_row_stencil, applyK_debug, applyB_debug,
                                       eval_sign_full, eval_J_full, out);
    default:
      return AKV_ERR_BADPARAM;
  }
}
