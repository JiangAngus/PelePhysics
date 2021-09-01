#include "reactor.H"

using namespace amrex;

#ifdef AMREX_USE_GPU
static int
Precond( amrex::Real tn,
         N_Vector u,
         N_Vector fu,
         booleantype jok,
         booleantype* jcurPtr,
         amrex::Real gamma,
         void* user_data)
{
  BL_PROFILE_VAR("Precond()", Precond);

  Real* u_d = N_VGetDeviceArrayPointer_Cuda(u);
  Real* udot_d = N_VGetDeviceArrayPointer_Cuda(fu);

  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  udata->gamma = gamma;
  
  // Get data out of udata to minimize global memory access
  auto ncells = udata->ncells_d;
  auto stream = udata->stream;
  auto nbThread = udata->nbThreads;
  auto nbBlocks = udata->nbBlocks;
  auto csr_val_d = udata->csr_val_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;
  auto cusolverHandle = udata->cusolverHandle;
  auto NNZ = udata->NNZ;
  auto descrA = udata->descrA;
  auto info = udata->info;

  BL_PROFILE_VAR("fKernelComputeAJ()", fKernelComputeAJ);
  if (jok) {
    const auto ec = Gpu::ExecutionConfig(ncells);
    launch_global<<<
      nbBlocks, nbThreads, ec.sharedMem, stream>>>(
      [=] AMREX_GPU_DEVICE() noexcept {
        for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
                 stride = blockDim.x * gridDim.x;
             icell < ncells; icell += stride) {
          fKernelComputeAJsys(icell, user_data, u_d, csr_val_d);
        }
      });
    *jcurPtr = SUNFALSE;
  } else {
    const auto ec = Gpu::ExecutionConfig(ncells);
    launch_global<<<
      nbBlocks, nbThreads, ec.sharedMem, stream>>>(
      [=] AMREX_GPU_DEVICE() noexcept {
        for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
                 stride = blockDim.x * gridDim.x;
             icell < ncells; icell += stride) {
          fKernelComputeallAJ(icell, user_data, u_d, csr_val_d);
        }
      });
    *jcurPtr = SUNTRUE;
  }
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  assert(cuda_status == cudaSuccess);
  BL_PROFILE_VAR_STOP(fKernelComputeAJ);

  BL_PROFILE_VAR("InfoBatched(inPrecond)", InfoBatched);
  size_t workspaceInBytes = 0;
  size_t internalDataInBytes = 0;
  cusolverStatus_t cuS_st = CUSOLVER_STATUS_SUCCESS;
  cuS_st = cusolverSpDcsrqrBufferInfoBatched(cusolverHandle,
                                             NUM_SPECIES + 1, NUM_SPECIES + 1,
                                             NNZ, descrA, csr_val_d, 
                                             csr_row_count_d, csr_col_index_d, 
                                             ncells, info,
                                             &internalDataInBytes,
                                             &workspaceInBytes);
  assert(cuS_st == CUSOLVER_STATUS_SUCCESS);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  BL_PROFILE_VAR_STOP(InfoBatched);

  BL_PROFILE_VAR_STOP(Precond);

  return (0);
}

static int
PSolve( amrex::Real tn,
        N_Vector u,
        N_Vector fu,
        N_Vector r,
        N_Vector z,
        amrex::Real gamma,
        amrex::Real delta,
        int lr,
        void* user_data)
{
  BL_PROFILE_VAR("Psolve()", cusolverPsolve);

  // Get data out of udata to minimize global memory access
  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells;
  auto csr_val_d = udata->csr_val_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;
  auto cusolverHandle = udata->cusolverHandle;
  auto NNZ = udata->NNZ;
  auto descrA = udata->descrA;
  auto info = udata->info;
  auto buffer_qr = udata->buffer_qr;
 
  Real* z_d = N_VGetDeviceArrayPointer_Cuda(z);
  Real* r_d = N_VGetDeviceArrayPointer_Cuda(r);

  cusolverStatus_t cuS_st = CUSOLVER_STATUS_SUCCESS;
  cuS_st = cusolverSpDcsrqrsvBatched( cusolverHandle, 
                                       NUM_SPECIES + 1, NUM_SPECIES + 1,
                                       NNZ, descrA, 
                                       csr_val_d, csr_row_count_d, 
                                       csr_col_index_d, r_d, z_d, 
                                       ncells, info, buffer_qr);
  assert(cuS_st == CUSOLVER_STATUS_SUCCESS);

  cudaError_t cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  N_VCopyFromDevice_Cuda(z);
  N_VCopyFromDevice_Cuda(r);

  BL_PROFILE_VAR_STOP(cusolverPsolve);

  // Checks
  // if (udata->iverbose > 4) {
  //    for(int batchId = 0 ; batchId < udata->ncells; batchId++){
  //        // measure |bj - Aj*xj|
  //        realtype *csrValAj = (udata->csr_val_d) + batchId * (udata->NNZ);
  //        double *xj       = N_VGetHostArrayPointer_Cuda(z) + batchId *
  //        (udata->neqs_per_cell+1); double *bj       =
  //        N_VGetHostArrayPointer_Cuda(r) + batchId * (udata->neqs_per_cell+1);
  //        // sup| bj - Aj*xj|
  //        double sup_res = 0;
  //        for(int row = 0 ; row < (udata->neqs_per_cell+1) ; row++){
  //            printf("\n     row %d: ", row);
  //            const int start = udata->csr_row_count_d[row] - 1;
  //            const int end = udata->csr_row_count_d[row +1] - 1;
  //            double Ax = 0.0; // Aj(row,:)*xj
  //            for(int colidx = start ; colidx < end ; colidx++){
  //                const int col = udata->csr_col_index_d[colidx] - 1;
  //                const double Areg = csrValAj[colidx];
  //                const double xreg = xj[col];
  //                printf("  (%d, %14.8e, %14.8e, %14.8e) ",
  //                col,Areg,xreg,bj[row] ); Ax = Ax + Areg * xreg;
  //            }
  //            double rresidi = bj[row] - Ax;
  //            sup_res = (sup_res > fabs(rresidi))? sup_res : fabs(rresidi);
  //        }
  //        printf("batchId %d: sup|bj - Aj*xj| = %E \n", batchId, sup_res);
  //    }
  //}

  return (0);
}

#else

// Preconditioner setup routine for GMRES solver when no sparse mode is
// activated Generate and preprocess P
int
Precond( Real /* tn */,
         N_Vector u,
         N_Vector /* fu */,
         booleantype jok,
         booleantype* jcurPtr,
         Real gamma,
         void* user_data)
{
  // Make local copies of pointers to input data
  Real* u_d = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells_d;
  auto reactor_type = udata->ireactor_type;
  auto P = udata->P;
  auto Jbd = udata->Jbd;
  auto pivot = udata->pivot;

  // MW CGS
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  if (jok) {
    // jok = SUNTRUE: Copy Jbd to P
    denseCopy(Jbd[0][0], P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);
    *jcurPtr = SUNFALSE;
  } else {
    // rho MKS
    Real rho = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      rho = rho + u_d[offset + i];
    }
    // Yks
    Real massfrac[NUM_SPECIES] = {0.0};
    Real rhoinv = 1.0/rho;
    for (int i = 0; i < NUM_SPECIES; i++) {
      massfrac[i] = u_d[offset + i] * rhoinv;
    }
    Real temp = u_d[offset + NUM_SPECIES];
    // Activities
    Real activity[NUM_SPECIES] = {0.0};
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2C(rho, temp, massfrac, activity);
    int consP = reactor_type == eint_rho ? 0 : 1;
    Real Jmat[(NUM_SPECIES + 1) * (NUM_SPECIES + 1)] = {0.0};
    DWDOT_SIMPLIFIED(Jmat, activity, &temp, &consP);

    // Scale Jacobian.  Load into P.
    denseScale(0.0, Jbd[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);
    for (int i = 0; i < NUM_SPECIES; i++) {
      for (int k = 0; k < NUM_SPECIES; k++) {
        (Jbd[0][0])[k][i] = Jmat[k * (NUM_SPECIES + 1) + i] * mw[i] / mw[k];
      }
      (Jbd[0][0])[i][NUM_SPECIES] =
        Jmat[i * (NUM_SPECIES + 1) + NUM_SPECIES] / mw[i];
    }
    for (int i = 0; i < NUM_SPECIES; i++) {
      (Jbd[0][0])[NUM_SPECIES][i] =
        Jmat[NUM_SPECIES * (NUM_SPECIES + 1) + i] * mw[i];
    }
    (Jbd[0][0])[NUM_SPECIES][NUM_SPECIES] =
      Jmat[(NUM_SPECIES + 1) * (NUM_SPECIES + 1) - 1];

    denseCopy(Jbd[0][0], P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);

    *jcurPtr = SUNTRUE;
  }

  // Scale by -gamma
  denseScale(-gamma, P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);
  // denseScale(0.0, P[0][0], NUM_SPECIES+1, NUM_SPECIES+1);

  // Add identity matrix and do LU decompositions on blocks in place.
  denseAddIdentity(P[0][0], NUM_SPECIES + 1);
  sunindextype ierr = denseGETRF(P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1, pivot[0][0]);
  if (ierr != 0)
    return (1);

  return (0);
}

int
PSolve( Real /* tn */,
        N_Vector /* u */,
        N_Vector /* fu */,
        N_Vector r,
        N_Vector z,
        Real /* gamma */,
        Real /* delta */,
        int /* lr */,
        void* user_data)
{
  // Make local copies of pointers to input data
  Real* zdata = N_VGetArrayPointer(z);

  // Extract the P and pivot arrays from user_data.
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto P = udata->P;
  auto pivot = udata->pivot;

  N_VScale(1.0, r, z);

  // Solve the block-diagonal system Pz = r using LU factors stored
  //   in P and pivot data in pivot, and return the solution in z.
  Real* v = zdata;
  denseGETRS(P[0][0], NUM_SPECIES + 1, pivot[0][0], v);

  return (0);
}

#ifdef USE_KLU_PP
// Preconditioner setup routine for GMRES solver when KLU sparse mode is
// activated Generate and preprocess P
int
Precond_sparse(
  realtype /* tn */,
  N_Vector u,
  N_Vector /* fu */,
  booleantype jok,
  booleantype* jcurPtr,
  realtype gamma,
  void* user_data)
{
  // Make local copies of pointers to input data (big M)
  realtype* udata = N_VGetArrayPointer(u);
  // Make local copies of pointers in user_data (cell M)
  CVODEUserData *data_wk = (CVODEUserData *)user_data;

  // MW CGS
  realtype molecular_weight[NUM_SPECIES];
  CKWT(molecular_weight);

  // Check if Jac is stale
  if (jok) {
    // jok = SUNTRUE: Copy Jbd to P
    *jcurPtr = SUNFALSE;
  } else {
    // Temp vectors
    realtype activity[NUM_SPECIES], massfrac[NUM_SPECIES];
    // Save Jac from cell to cell if more than one
    realtype temp_save_lcl = 0.0;
    for (int tid = 0; tid < data_wk->ncells; tid++) {
      // Offset in case several cells
      int offset = tid * (NUM_SPECIES + 1);
      // rho MKS
      realtype rho = 0.0;
      for (int i = 0; i < NUM_SPECIES; i++) {
        rho = rho + udata[offset + i];
      }
      // Yks
      for (int i = 0; i < NUM_SPECIES; i++) {
        massfrac[i] = udata[offset + i] / rho;
      }
      realtype temp = udata[offset + NUM_SPECIES];
      // Activities
      auto eos = pele::physics::PhysicsType::eos();
      eos.RTY2C(rho, temp, massfrac, activity);
      // Do we recompute Jac ?
      if (fabs(temp - temp_save_lcl) > 1.0) {
        // Formalism
        int consP = data_wk->ireactor_type == eint_rho ? 0 : 1;
        DWDOT_SIMPLIFIED(data_wk->JSPSmat[tid], activity, &temp, &consP);

        for (int i = 0; i < NUM_SPECIES; i++) {
          for (int k = 0; k < NUM_SPECIES; k++) {
            (data_wk->JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] =
              (data_wk->JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] *
              molecular_weight[i] / molecular_weight[k];
          }
          (data_wk->JSPSmat[tid])[i * (NUM_SPECIES + 1) + NUM_SPECIES] =
            (data_wk->JSPSmat[tid])[i * (NUM_SPECIES + 1) + NUM_SPECIES] /
            molecular_weight[i];
        }
        for (int i = 0; i < NUM_SPECIES; i++) {
          (data_wk->JSPSmat[tid])[NUM_SPECIES * (NUM_SPECIES + 1) + i] =
            (data_wk->JSPSmat[tid])[NUM_SPECIES * (NUM_SPECIES + 1) + i] *
            molecular_weight[i];
        }
        temp_save_lcl = temp;
      } else {
        // if not: copy the one from prev cell
        for (int i = 0; i < NUM_SPECIES + 1; i++) {
          for (int k = 0; k < NUM_SPECIES + 1; k++) {
            (data_wk->JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] =
              (data_wk->JSPSmat[tid - 1])[k * (NUM_SPECIES + 1) + i];
          }
        }
      }
    }

    *jcurPtr = SUNTRUE;
  }

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    // nb non zeros elem should be the same for all cells
    int nbVals = data_wk->colPtrs[0][i] - data_wk->colPtrs[0][i - 1];
    for (int j = 0; j < nbVals; j++) {
      // row of non zero elem should be the same for all cells
      int idx = data_wk->rowVals[0][data_wk->colPtrs[0][i - 1] + j];
      // Scale by -gamma
      // Add identity matrix
      for (int tid = 0; tid < data_wk->ncells; tid++) {
        if (idx == (i - 1)) {
          data_wk->Jdata[tid][data_wk->colPtrs[tid][i - 1] + j] =
            1.0 -
            gamma * (data_wk->JSPSmat[tid])[idx * (NUM_SPECIES + 1) + idx];
        } else {
          data_wk->Jdata[tid][data_wk->colPtrs[tid][i - 1] + j] =
            -gamma * (data_wk->JSPSmat[tid])[(i - 1) * (NUM_SPECIES + 1) + idx];
        }
      }
    }
  }

  BL_PROFILE_VAR("KLU_factorization", KLU_factor);
  if (!(data_wk->FirstTimePrecond)) {
    for (int tid = 0; tid < data_wk->ncells; tid++) {
      klu_refactor(
        data_wk->colPtrs[tid], data_wk->rowVals[tid], data_wk->Jdata[tid],
        data_wk->Symbolic[tid], data_wk->Numeric[tid], &(data_wk->Common[tid]));
    }
  } else {
    for (int tid = 0; tid < data_wk->ncells; tid++) {
      data_wk->Numeric[tid] = klu_factor(
        data_wk->colPtrs[tid], data_wk->rowVals[tid], data_wk->Jdata[tid],
        data_wk->Symbolic[tid], &(data_wk->Common[tid]));
    }
    data_wk->FirstTimePrecond = false;
  }
  BL_PROFILE_VAR_STOP(KLU_factor);

  return (0);
}

int
PSolve_sparse(Real /* tn */,
              N_Vector /* u */,
              N_Vector /* fu */,
              N_Vector r,
              N_Vector z,
              Real /* gamma */,
              Real /* delta */,
              int /* lr */,
              void* user_data)
{
  // Make local copies of pointers in user_data
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells_d;
  auto Symbolic = udata->Symbolic;
  auto Numeric  = udata->Numeric;
  auto Common   = udata->Common;

  // Make local copies of pointers to input data (big M)
  Real* zdata = N_VGetArrayPointer(z);

  BL_PROFILE_VAR("KLU_inversion", PSolve_sparse);
  N_VScale(1.0, r, z);

  // Solve the block-diagonal system Pz = r using LU factors stored
  //   in P and pivot data in pivot, and return the solution in z.
  Real zdata_cell[NUM_SPECIES + 1];
  for (int tid = 0; tid < ncells; tid++) {
    int offset_beg = tid * (NUM_SPECIES + 1);
    std::memcpy(
      zdata_cell, zdata + offset_beg, (NUM_SPECIES + 1) * sizeof(Real));
    klu_solve(
      Symbolic[tid], Numeric[tid], NUM_SPECIES + 1, 1,
      zdata_cell, &(Common[tid]));
    std::memcpy(
      zdata + offset_beg, zdata_cell, (NUM_SPECIES + 1) * sizeof(Real));
  }
  BL_PROFILE_VAR_STOP(PSolve_sparse);

  return (0);
}
#endif

// Preconditioner setup routine for GMRES solver when custom sparse mode is
// activated Generate and preprocess P
int
Precond_custom( Real /* tn */,
                N_Vector u,
                N_Vector /* fu */,
                booleantype jok,
                booleantype* jcurPtr,
                Real gamma,
                void* user_data)
{
  // Make local copies of pointers to input data
  Real *u_d = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells_d;
  auto reactor_type = udata->ireactor_type;
  auto JSPSmat = udata->JSPSmat;
  auto rowPtrs = udata->rowPtrs;
  auto colVals = udata->colVals;
  auto Jdata = udata->Jdata;

  // MW CGS
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  // Check if Jac is stale
  if (jok) {
    // jok = SUNTRUE: Copy Jbd to P
    *jcurPtr = SUNFALSE;
  } else {
    // Save Jac from cell to cell if more than one
    Real temp_save_lcl = 0.0;
    for (int tid = 0; tid < ncells; tid++) {
      // Offset in case several cells
      int offset = tid * (NUM_SPECIES + 1);
      // rho MKS
      Real rho = 0.0;
      for (int i = 0; i < NUM_SPECIES; i++) {
        rho = rho + u_d[offset + i];
      }
      // Yks
      Real massfrac[NUM_SPECIES] = {0.0};
      Real rhoinv = 1.0/rho;
      for (int i = 0; i < NUM_SPECIES; i++) {
        massfrac[i] = u_d[offset + i] * rhoinv;
      }
      Real temp = u_d[offset + NUM_SPECIES];
      // Activities
      Real activity[NUM_SPECIES] = {0.0};
      auto eos = pele::physics::PhysicsType::eos();
      eos.RTY2C(rho, temp, massfrac, activity);

      // Do we recompute Jac ?
      if (fabs(temp - temp_save_lcl) > 1.0) {
        // Formalism
        int consP = reactor_type == eint_rho ? 0 : 1;
        DWDOT_SIMPLIFIED(JSPSmat[tid], activity, &temp, &consP);

        for (int i = 0; i < NUM_SPECIES; i++) {
          for (int k = 0; k < NUM_SPECIES; k++) {
            (JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] *= mw[i] / mw[k];
          }
          (JSPSmat[tid])[i * (NUM_SPECIES + 1) + NUM_SPECIES] /= mw[i];
        }
        for (int i = 0; i < NUM_SPECIES; i++) {
          (JSPSmat[tid])[NUM_SPECIES * (NUM_SPECIES + 1) + i] *= mw[i];
        }
        temp_save_lcl = temp;
      } else {
        // if not: copy the one from prev cell
        for (int i = 0; i < NUM_SPECIES + 1; i++) {
          for (int k = 0; k < NUM_SPECIES + 1; k++) {
            (JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] = (JSPSmat[tid - 1])[k * (NUM_SPECIES + 1) + i];
          }
        }
      }
    }
    *jcurPtr = SUNTRUE;
  }

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    // nb non zeros elem should be the same for all cells
    int nbVals = rowPtrs[0][i] - rowPtrs[0][i - 1];
    for (int j = 0; j < nbVals; j++) {
      // row of non zero elem should be the same for all cells
      int idx = colVals[0][rowPtrs[0][i - 1] + j];
      // Scale by -gamma
      // Add identity matrix
      for (int tid = 0; tid < ncells; tid++) {
        if (idx == (i - 1)) {
          Jdata[tid][rowPtrs[tid][i - 1] + j] = 1.0 - gamma * (JSPSmat[tid])[idx * (NUM_SPECIES + 1) + idx];
        } else {
          Jdata[tid][rowPtrs[tid][i - 1] + j] = -gamma * (JSPSmat[tid])[(i - 1) + (NUM_SPECIES + 1) * idx];
        }
      }
    }
  }

  return (0);
}

int
PSolve_custom( Real /* tn */,
               N_Vector /* u */,
               N_Vector /* fu */,
               N_Vector r,
               N_Vector z,
               Real /* gamma */,
               Real /* delta */,
               int /* lr */,
               void* user_data)
{
  // Make local copies of pointers in user_data
  CVODEUserData *udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells_d;
  auto Jdata = udata->Jdata;

  // Make local copies of pointers to input data
  Real* zdata = N_VGetArrayPointer(z);
  Real* rdata = N_VGetArrayPointer(r);

  N_VScale(1.0, r, z);

  // Solve the block-diagonal system Pz = r using LU factors stored
  // in P and pivot data in pivot, and return the solution in z.
  BL_PROFILE_VAR("GaussSolver", GaussSolver);
  for (int tid = 0; tid < ncells; tid++) {
    int offset = tid * (NUM_SPECIES + 1);
    double* z_d_offset = zdata + offset;
    double* r_d_offset = rdata + offset;
    sgjsolve_simplified(Jdata[tid], z_d_offset, r_d_offset);
  }
  BL_PROFILE_VAR_STOP(GaussSolver);

  return (0);
}

#endif
