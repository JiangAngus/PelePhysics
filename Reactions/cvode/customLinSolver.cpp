#include "reactor.H"

using namespace amrex;

#ifdef AMREX_USE_GPU

// The following are only available with Cuda
#ifdef AMREX_USE_CUDA
SUNLinearSolver
SUNLinSol_dense_custom(N_Vector y, 
                       SUNMatrix A,
                       cudaStream_t stream)
{
  if (y == NULL || A == NULL)
    return (NULL);

  if (N_VGetVectorID(y) != SUNDIALS_NVEC_CUDA)
    return (NULL);
  if (SUNMatGetID(A) != SUNMATRIX_CUSPARSE)
    return (NULL);

  if (N_VGetLength(y) != SUNMatrix_cuSparse_Columns(A))
    return (NULL);

  if (!N_VIsManagedMemory_Cuda(y))
    return (NULL);

  SUNLinearSolver S;
  S = NULL;
  S = SUNLinSolNewEmpty();
  if (S == NULL) {
    return (NULL);
  }

  S->ops->gettype = SUNLinSolGetType_Dense_custom;
  S->ops->setup = SUNLinSolSetup_Dense_custom;
  S->ops->solve = SUNLinSolSolve_Dense_custom;
  S->ops->free = SUNLinSolFree_Dense_custom;

  SUNLinearSolverContent_Dense_custom content;
  content = NULL;
  content = (SUNLinearSolverContent_Dense_custom)malloc(sizeof *content);
  if (content == NULL) {
    SUNLinSolFree(S);
    return (NULL);
  }

  S->content = content;

  content->last_flag = 0;
  content->nsubsys = SUNMatrix_cuSparse_NumBlocks(A);
  content->subsys_size = SUNMatrix_cuSparse_BlockRows(A);
  content->subsys_nnz = SUNMatrix_cuSparse_BlockNNZ(A);
  content->nbBlocks = std::max(1, content->nsubsys / 32);
  content->nbThreads = 32;
  content->stream = stream;

  return (S);
}

SUNLinearSolver_Type
SUNLinSolGetType_Dense_custom(SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_DIRECT);
}

int
SUNLinSolSetup_Dense_custom(SUNLinearSolver S, SUNMatrix A)
{
  return (SUNLS_SUCCESS);
}

int
SUNLinSolSolve_Dense_custom(SUNLinearSolver S,
                            SUNMatrix A,
                            N_Vector x,
                            N_Vector b,
                            Real tol)
{
  cudaError_t cuda_status = cudaSuccess;

  Real* x_d = N_VGetDeviceArrayPointer_Cuda(x);
  Real* b_d = N_VGetDeviceArrayPointer_Cuda(b);

  Real* d_data = SUNMatrix_cuSparse_Data(A);

  BL_PROFILE_VAR("fKernelDenseSolve()", fKernelDenseSolve);
  const auto ec = Gpu::ExecutionConfig(SUN_CUSP_NUM_SUBSYS(S));
  // TODO: why is this AMREX version NOT working ?
  // launch_global<<<ec.numBlocks, ec.numThreads, ec.sharedMem,
  // SUN_CUSP_STREAM(S)>>>(
  //    [=] AMREX_GPU_DEVICE () noexcept {
  //        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride =
  //        blockDim.x*gridDim.x;
  //            icell < SUN_CUSP_NUM_SUBSYS(S); icell += stride) {
  //            fKernelDenseSolve(icell, x_d, b_d,
  //                  SUN_CUSP_SUBSYS_SIZE(S), SUN_CUSP_SUBSYS_NNZ(S), data_d);
  //        }
  //    });
  fKernelDenseSolve<<<
    SUN_CUSP_NBLOCK(S), SUN_CUSP_NTHREAD(S), ec.sharedMem,
    SUN_CUSP_STREAM(S)>>>(
    SUN_CUSP_NUM_SUBSYS(S), x_d, b_d, SUN_CUSP_SUBSYS_SIZE(S),
    SUN_CUSP_SUBSYS_NNZ(S), d_data);

  cuda_status = cudaStreamSynchronize(SUN_CUSP_STREAM(S));
  assert(cuda_status == cudaSuccess);

  BL_PROFILE_VAR_STOP(fKernelDenseSolve);

  return (SUNLS_SUCCESS);
}

int
SUNLinSolFree_Dense_custom(SUNLinearSolver S)
{
  if (S == NULL)
    return (SUNLS_SUCCESS);

  if (S->content) {
    free(S->content);
    S->content = NULL;
  }

  if (S->ops) {
    free(S->ops);
    S->ops = NULL;
  }

  free(S);
  S = NULL;

  return (SUNLS_SUCCESS);
}
#endif

#else

SUNLinearSolver
SUNLinSol_sparse_custom( N_Vector a_y,
                         SUNMatrix a_A,
                         int reactor_type,
                         int nsubsys,
                         int subsys_size,
                         int subsys_nnz)
{
  SUNLinearSolver S;
  SUNLinearSolverContent_Sparse_custom content;

  // Check that required arguments are not NULL
  if (a_y == NULL || a_A == NULL) return (NULL);
  if (SUNMatGetID(a_A) != SUNMATRIX_SPARSE) return (NULL);

  // Matrix should be square
  if (SUNSparseMatrix_Columns(a_A) != SUNSparseMatrix_Rows(a_A))
    return (NULL);

  // Check that it is a CSR matrix
  if (SUNSparseMatrix_SparseType(a_A) != CSR_MAT)
    return (NULL);

  // Matrix and vector dimensions must agree
  if (N_VGetLength(a_y) != SUNSparseMatrix_Columns(a_A))
    return (NULL);

  // All subsystems must be the same size
  if (SUNSparseMatrix_Columns(a_A) != (subsys_size * nsubsys))
    return (NULL);

  // Number of nonzeros per subsys must be the same
  if (SUNSparseMatrix_NNZ(a_A) != (subsys_nnz * nsubsys))
    return (NULL);

  // Create an empty linear solver
  S = SUNLinSolNewEmpty();
  if (S == NULL)
    return (NULL);

  // Attach operations
  S->ops->gettype = SUNLinSolGetType_Sparse_custom;
  S->ops->solve = SUNLinSolSolve_Sparse_custom;

  // Create content
  content = (SUNLinearSolverContent_Sparse_custom)malloc(sizeof *content);
  if (content == NULL) {
    SUNLinSolFree(S);
    return (NULL);
  }

  // Attach content
  S->content = content;

  // Fill content
  content->last_flag = 0;
  content->reactor_type = reactor_type;
  content->nsubsys = nsubsys;
  content->subsys_size = subsys_size;
  content->subsys_nnz = subsys_nnz;

  return (S);
}

SUNLinearSolver_Type SUNLinSolGetType_Sparse_custom(SUNLinearSolver /* S */)
{
  return (SUNLINEARSOLVER_DIRECT);
}

int
SUNLinSolSolve_Sparse_custom( SUNLinearSolver S,
                              SUNMatrix a_A,
                              N_Vector x,
                              N_Vector b,
                              Real /*tol*/)
{
  Real* x_d = N_VGetArrayPointer(x);
  Real* b_d = N_VGetArrayPointer(b);

  Real* Data = (Real*)SUNSparseMatrix_Data(a_A);

  BL_PROFILE_VAR("GaussSolver", GaussSolver);
  for (int tid = 0; tid < SUN_CUSP_NUM_SUBSYS(S); tid++) {
    int offset = tid * SUN_CUSP_SUBSYS_NNZ(S);
    int offset_RHS = tid * SUN_CUSP_SUBSYS_SIZE(S);
    Real* Data_offset = Data + offset;
    Real* x_d_offset = x_d + offset_RHS;
    Real* b_d_offset = b_d + offset_RHS;
    sgjsolve(Data_offset, x_d_offset, b_d_offset);
  }
 RealOFILE_VAR_STOP(GaussSolver);

  return (SUNLS_SUCCESS);
}
#endif