#include "reactor.H"

using namespace amrex;

#ifdef AMREX_USE_GPU

// Will not work for cuSolver_sparse_solve right now
static int
cJac( amrex::Real /*t*/,
      N_Vector y_in,
      N_Vector /*fy*/,
      SUNMatrix J,
      void* user_data,
      N_Vector /*tmp1*/,
      N_Vector /*tmp2*/,
      N_Vector /*tmp3*/)
{
  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells_d;
  auto NNZ = udata->NNZ;
  auto stream = udata->stream;
  auto nbThread = udata->nbThreads;
  auto nbBlocks = udata->nbBlocks;

  Real* yvec_d = N_VGetDeviceArrayPointer_Cuda(y_in);

  Real* Jdata = SUNMatrix_cuSparse_Data(J);

  // Checks
  if ( ( SUNMatrix_cuSparse_Rows(J)    != (NUM_SPECIES + 1) * ncells) ||
       ( SUNMatrix_cuSparse_Columns(J) != (NUM_SPECIES + 1) * ncells) ||
       ( SUNMatrix_cuSparse_NNZ(J)     != ncells * NNZ)) {
    Print() << "Jac error: matrix is wrong size!\n";
    return 1;
  }

  BL_PROFILE_VAR("Jacobian()", fKernelJac);
  const auto ec = Gpu::ExecutionConfig(ncells);
  launch_global<<<nbBlocks, nbThreads, ec.sharedMem, stream>>>(
    [=] AMREX_GPU_DEVICE() noexcept {
      for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
               stride = blockDim.x * gridDim.x;
           icell < ncells; icell += stride) {
        fKernelComputeAJchem(icell, user_data, yvec_d, Jdata);
      }
    });
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  assert(cuda_status == cudaSuccess);
  BL_PROFILE_VAR_STOP(fKernelJac);

  return (0);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeAJchem(int ncell,
                     void* user_data,
                     Real* u_d,
                     Real* Jdata)
{
  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  int neqs = NUM_SPECIES+1;
  auto NNZ = udata->NNZ;
  auto reactor_type = udata->ireactor_type;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;

  int u_offset = ncell * neqs;
  Real* u_curr = u_d + u_offset;

  Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rho_pt = rho_pt + u_curr[n];
  }

  GpuArray<Real, NUM_SPECIES> massfrac = {0.0};
  for (int n = 0; n < NUM_SPECIES; n++) {
    massfrac[n] = u_curr[n] / rho_pt;
  }
  Real temp_pt = u_curr[NUM_SPECIES];

  int consP;
  if (reactor_type == 1) {
    consP = 0;
  } else {
    consP = 1;
  }
  GpuArray<Real, neqs * neqs> Jmat_pt = {0.0};
  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2JAC(rho_pt, temp_pt, massfrac.arr, Jmat_pt.arr, consP);

  // Scale Jacobian and pass into outgoing data ptr.
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  int jac_offset = ncell * NNZ;
  Real* csr_jac_cell = Jdata + jac_offset;

  for (int i = 0; i < NUM_SPECIES; i++) {
    for (int k = 0; k < NUM_SPCIES; k++) {
      Jmat_pt[k * neqs + i] *= mw[i] / mw[k];
    }
    Jmat_pt[i * neqs + NUM_SPECIES] /= mw[i];
    Jmat_pt[NUM_SPECIES * neqs + i] *= mw[i];
  }
  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    int nbVals = csr_row_count_d[i] - csr_row_count_d[i - 1];
    for (int j = 0; j < nbVals; j++) {
      int idx_cell = csr_col_index_d[csr_row_count_d[i - 1] + j];
      csr_jac_cell[csr_row_count_d[i - 1] + j] = Jmat_pt[idx_cell * neqs + i - 1];
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeallAJ(int ncell,
                    void* user_data, 
                    Real* u_d,
                    Real* csr_val_arg)
{
  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  int neqs = NUM_SPECIES+1;
  auto NNZ = udata->NNZ;
  auto reactor_type = udata->ireactor_type;
  auto csr_jac_d = udata->csr_jac_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;
  auto gamma = udata->gamma;

  GpuArray<Real, NUM_SPECIES> activity{0.0};
  GpuArray<Real, (NUM_SPECIES + 1) * (NUM_SPECIES + 1)> Jmat_pt{0.0};


  int u_offset = ncell * neqs;
  Real *u_curr = u_d + u_offset;

  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rho_pt = rho_pt + u_curr[n];
  }

  GpuArray<Real, NUM_SPECIES> massfrac{0.0};
  for (int i = 0; i < NUM_SPECIES; i++) {
    massfrac[i] = u_curr[i] / rho_pt;
  }

  Real temp_pt = u_curr[NUM_SPECIES];

  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2C(rho_pt, temp_pt, massfrac.arr, activity.arr);

  int consP;
  if (udata->ireactor_type == 1) {
    consP = 0;
  } else {
    consP = 1;
  }
  DWDOT_SIMPLIFIED(Jmat_pt.arr, activity.arr, &temp_pt, &consP);

  int jac_offset = ncell * NNZ;
  Real* csr_jac_cell = csr_jac_d + jac_offset;
  Real* csr_val_cell = csr_val_arg + jac_offset;
  for (int i = 0; i < NUM_SPECIES; i++) {
    for (int k = 0; k < NUM_SPECIES; k++) {
      Jmat_pt[k * neqs + i] *= mw[i] / mw[k];
    }
    Jmat_pt[i * neqs + NUM_SPECIES] /= mw[i]; 
    Jmat_pt[NUM_SPECIES * neqs + i] *= mw[i];
  }
  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    int nbVals = csr_row_count_d[i] - csr_row_count_d[i - 1];
    for (int j = 0; j < nbVals; j++) {
      int idx = csr_col_index_d[csr_row_count_d[i - 1] + j - 1] - 1;
      if (idx == (i - 1)) { 
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] = 1.0 - gamma * Jmat_pt[idx * neqs + idx];
        csr_jac_cell[csr_row_count_d[i - 1] + j - 1] = Jmat_pt[idx * neqs + idx];
      } else {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] = -gamma * Jmat_pt[idx * neqs + i - 1];
        csr_jac_cell[csr_row_count_d[i - 1] + j - 1] = Jmat_pt[idx * neqs + i - 1];
      }
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeAJsys(int ncell,
                    void* user_data,
                    Real* u_d,
                    Real* csr_val_arg)
{
  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  auto NNZ = udata->NNZ;
  auto csr_jac_d = udata->csr_jac_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;
  auto gamma = udata->gamma;

  int jac_offset = ncell * NNZ;
  Real* csr_jac_cell = csr_jac_d + jac_offset;
  Real* csr_val_cell = csr_val_arg + jac_offset;

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    int nbVals = csr_row_count_d[i] - csr_row_count_d[i - 1];
    for (int j = 0; j < nbVals; j++) {
      int idx =csr_col_index_d[csr_row_count_d[i - 1] + j - 1] - 1;
      if (idx == (i - 1)) {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] = 1.0 - gamma * csr_jac_cell[csr_row_count_d[i - 1] + j - 1];
      } else {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] = -gamma * csr_jac_cell[csr_row_count_d[i - 1] + j - 1];
      }
    }
  }
}

#else

int
cJac( Real /* tn */,
      N_Vector u,
      N_Vector /* fu */,
      SUNMatrix J,
      void* user_data,
      N_Vector /* tmp1 */,
      N_Vector /* tmp2 */,
      N_Vector /* tmp3 */)
{

  // Make local copies of pointers to input data
  Real* ydata = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data
  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells_d;
  auto reactor_type = udata->ireactor_type;

  BL_PROFILE_VAR("DenseJac", DenseJac);
  for (int tid = 0; tid < ncells; tid++) {
    // Offset in case several cells
    int offset = tid * (NUM_SPECIES + 1);

    // MW CGS
    Real mw[NUM_SPECIES] = {0.0};
    get_mw(mw);

    // rho MKS
    Real rho = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      rho = rho + ydata[offset + i];
    }

    Real temp = ydata[offset + NUM_SPECIES];

    Real massfrac[NUM_SPECIES] = {0.0};
    // Yks
    for (int i = 0; i < NUM_SPECIES; i++) {
      massfrac[i] = ydata[offset + i] / rho;
    }

    // Jac
    Real Jmat_tmp[(NUM_SPECIES + 1) * (NUM_SPECIES + 1)] = {0.0};
    int consP = reactor_type == eint_rho ? 0 : 1;
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2JAC(rho, temp, massfrac, Jmat_tmp, consP);

    // fill the sunMat and scale
    for (int k = 0; k < NUM_SPECIES; k++) {
      Real *J_col_k = SM_COLUMN_D(J, offset + k);
      for (int i = 0; i < NUM_SPECIES; i++) {
        J_col_k[offset + i] = Jmat_tmp[k * (NUM_SPECIES + 1) + i] *
                              mw[i] / mw[k];
      }
      J_col_k[offset + NUM_SPECIES] = Jmat_tmp[k * (NUM_SPECIES + 1) + NUM_SPECIES] / mw[k];
    }
    J_col_k = SM_COLUMN_D(J, offset + NUM_SPECIES);
    for (int i = 0; i < NUM_SPECIES; i++) {
      J_col_k[offset + i] = Jmat_tmp[NUM_SPECIES * (NUM_SPECIES + 1) + i] * mw[i];
    }
    J_col_k = SM_COLUMN_D(J, offset);
  }
  BL_PROFILE_VAR_STOP(DenseJac);

  return (0);
}

// Analytical SPARSE CSR Jacobian evaluation
int
cJac_sps( Reql /* tn */,
          N_Vector u,
          N_Vector /* fu */,
          SUNMatrix J,
          void* user_data,
          N_Vector /* tmp1 */,
          N_Vector /* tmp2 */,
          N_Vector /* tmp3 */)
{
  // Make local copies of pointers to input data
  Real* ydata = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data (cell M)*/
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto NNZ = udata->NNZ;
  auto reactor_type = udata->ireactor_type;
  auto ncells = udata->ncells_d;
  auto colVals_c = udata->colVals_c;
  auto rowPtrs_c = udata->rowPtrs_c;

  // MW CGS
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  BL_PROFILE_VAR("FillSparseData", FillSpsData);
  sunindextype* rowPtrs_tmp = SUNSparseMatrix_IndexPointers(J);
  sunindextype* colIndx_tmp = SUNSparseMatrix_IndexValues(J);
  Real* Jdata = SUNSparseMatrix_Data(J);
  // Fixed colVal
  for (int i = 0; i < NNZ * ncells; i++) {
    colIndx_tmp[i] = (sunindextype)colVals_c[i];
  }
  rowPtrs_tmp[0] = (sunindextype)rowPtrs_c[0];
  // Fixed rowPtrs
  for (int i = 0; i < ncells * (NUM_SPECIES + 1); i++) {
    rowPtrs_tmp[i + 1] = (sunindextype)rowPtrs_c[i + 1];
  }
  BL_PROFILE_VAR_STOP(FillSpsData);

  BL_PROFILE_VAR("SparseJac", SpsJac);
  // Temp vectors
  // Save Jac from cell to cell if more than one
  Real temp_save_lcl = 0.0;
  for (int tid = 0; tid < ncells; tid++) {
    // Offset in case several cells
    int offset = tid * (NUM_SPECIES + 1);
    int offset_J = tid * NNZ;
    // rho MKS
    Real rho = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      rho = rho + ydata[offset + i];
    }
    // Yks
    Real massfrac[NUM_SPECIES] = {0.0};
    Real rhoinv = 1.0 / rho;
    for (int i = 0; i < NUM_SPECIES; i++) {
      massfrac[i] = ydata[offset + i] * rhoinv;
    }
    Real temp = ydata[offset + NUM_SPECIES];

    // Do we recompute Jac ?
    Real Jmat_tmp[(NUM_SPECIES + 1) * (NUM_SPECIES + 1)] = {0.0};
    if (fabs(temp - temp_save_lcl) > 1.0) {
      int consP = reactor_type == eint_rho ? 0 : 1;
      auto eos = pele::physics::PhysicsType::eos();
      eos.RTY2JAC(rho, temp, massfrac, Jmat_tmp, consP);
      temp_save_lcl = temp;
      // rescale
      for (int i = 0; i < NUM_SPECIES; i++) {
        for (int k = 0; k < NUM_SPECIES; k++) {
          Jmat_tmp[k * (NUM_SPECIES + 1) + i] *= mw[i] / mw[k];
        }
        Jmat_tmp[i * (NUM_SPECIES + 1) + NUM_SPECIES] /= mw[i];
      }
      for (int i = 0; i < NUM_SPECIES; i++) {
        Jmat_tmp[NUM_SPECIES * (NUM_SPECIES + 1) + i] *=  mw[i];
      }
    }
    // Go from Dense to Sparse
    for (int i = 1; i < NUM_SPECIES + 2; i++) {
      int nbVals = rowPtrs_c[i] - rowPtrs_c[i - 1];
      for (int j = 0; j < nbVals; j++) {
        int idx = colVals_c[rowPtrs_c[i - 1] + j];
        Jdata[offset_J + rowPtrs_c[i - 1] + j] =
          Jmat_tmp[(i - 1) + (NUM_SPECIES + 1) * idx];
      }
    }
  }
  BL_PROFILE_VAR_STOP(SpsJac);

  return (0);
}

#ifdef USE_KLU_PP
// Analytical SPARSE KLU CSC Jacobian evaluation
int
cJac_KLU( Real /* tn */,
          N_Vector u,
          N_Vector /* fu */,
          SUNMatrix J,
          void* user_data,
          N_Vector /* tmp1 */,
          N_Vector /* tmp2 */,
          N_Vector /* tmp3 */)
{
  BL_PROFILE_VAR("SparseKLUJac", SpsKLUJac);

  // Make local copies of pointers to input data
  Real* ydata = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data (cell M)
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto NNZ = udata->NNZ;
  auto reactor_type = udata->ireactor_type;
  auto ncells = udata->ncells_d;
  auto colPtrs = udata->colPtrs;
  auto rowVals = udata->rowVals;

  // MW CGS
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  // Fixed RowVals
  sunindextype* colptrs_tmp = SUNSparseMatrix_IndexPointers(J);
  sunindextype* rowvals_tmp = SUNSparseMatrix_IndexValues(J);
  Real* Jdata = SUNSparseMatrix_Data(J);
  for (int i = 0; i < NNZ; i++) {
    rowvals_tmp[i] = rowVals[0][i];
  }
  // Fixed colPtrs
  colptrs_tmp[0] = colPtrs[0][0];
  for (int i = 0; i < ncells * (NUM_SPECIES + 1); i++) {
    colptrs_tmp[i + 1] = colPtrs[0][i + 1];
  }

  // Save Jac from cell to cell if more than one
  Real temp_save_lcl = 0.0;
  for (int tid = 0; tid < ncells; tid++) {
    // Offset in case several cells
    int offset = tid * (NUM_SPECIES + 1);
    // rho
    Real rho = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      rho = rho + ydata[offset + i];
    }
    // Yks
    Real massfrac[NUM_SPECIES] = {0.0};
    Real rhoinv = 1.0 / rho;
    for (int i = 0; i < NUM_SPECIES; i++) {
      massfrac[i] = ydata[offset + i] * rhoinv;
    }
    Real temp = ydata[offset + NUM_SPECIES];

    // Do we recompute Jac ?
    if (fabs(temp - temp_save_lcl) > 1.0) {
      // NRG CGS
      int consP = reactor_type == eint_rho ? 0 : 1;
      auto eos = pele::physics::PhysicsType::eos();
      eos.RTY2JAC(rho, temp, massfrac, Jmat_tmp, consP);
      temp_save_lcl = temp;
      // rescale
      for (int i = 0; i < NUM_SPECIES; i++) {
        for (int k = 0; k < NUM_SPECIES; k++) {
          Jmat_tmp[k * (NUM_SPECIES + 1) + i] *= mw[i] / mw[k];
        }
        Jmat_tmp[i * (NUM_SPECIES + 1) + NUM_SPECIES] /= mw[i];
      }
      for (int i = 0; i < NUM_SPECIES; i++) {
        Jmat_tmp[NUM_SPECIES * (NUM_SPECIES + 1) + i] *= mw[i];
      }
    }
    // Go from Dense to Sparse
    BL_PROFILE_VAR("DensetoSps", DtoS);
    for (int i = 1; i < NUM_SPECIES + 2; i++) {
      int nbVals = colPtrs[0][i] - colPtrs[0][i - 1];
      for (int j = 0; j < nbVals; j++) {
        int idx = rowVals[0][colPtrs[0][i - 1] + j];
        Jdata[colPtrs[0][offset + i - 1] + j] =
          Jmat_tmp[(i - 1) * (NUM_SPECIES + 1) + idx];
      }
    }
    BL_PROFILE_VAR_STOP(DtoS);
  }

  BL_PROFILE_VAR_STOP(SpsKLUJac);

  return (0);
}
#endif

#endif
