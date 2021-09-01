#include "reactor.H"

using namespace amrex;

#ifdef AMREX_USE_GPU

#define SUN_CUSP_CONTENT(S) ((SUNLinearSolverContent_Dense_custom)(S->content))
#define SUN_CUSP_SUBSYS_SIZE(S) (SUN_CUSP_CONTENT(S)->subsys_size)
#define SUN_CUSP_NUM_SUBSYS(S) (SUN_CUSP_CONTENT(S)->nsubsys)
#define SUN_CUSP_SUBSYS_NNZ(S) (SUN_CUSP_CONTENT(S)->subsys_nnz)

#define SUN_CUSP_LASTFLAG(S) (SUN_CUSP_CONTENT(S)->last_flag)
#define SUN_CUSP_STREAM(S) (SUN_CUSP_CONTENT(S)->stream)
#define SUN_CUSP_NBLOCK(S) (SUN_CUSP_CONTENT(S)->nbBlocks)
#define SUN_CUSP_NTHREAD(S) (SUN_CUSP_CONTENT(S)->nbThreads)

#else

#define SUN_CUSP_CONTENT(S) ((SUNLinearSolverContent_Sparse_custom)(S->content))
#define SUN_CUSP_REACTYPE(S) (SUN_CUSP_CONTENT(S)->reactor_type)
#define SUN_CUSP_NUM_SUBSYS(S) (SUN_CUSP_CONTENT(S)->nsubsys)
#define SUN_CUSP_SUBSYS_NNZ(S) (SUN_CUSP_CONTENT(S)->subsys_nnz)
#define SUN_CUSP_SUBSYS_SIZE(S) (SUN_CUSP_CONTENT(S)->subsys_size)

// TODO: these will be moved to the cvodeReactor class later on
N_Vector y = NULL;
SUNLinearSolver LS = NULL;
SUNMatrix A = NULL;
void* cvode_mem = NULL;
CVODEUserData *user_data = NULL;

#ifdef AMREX_USE_OMP
#pragma omp threadprivate(y, LS, A)
#pragma omp threadprivate(cvode_mem, user_data)
#endif

#endif

// Error function for CVODE
void cvodeErrHandler(
  int error_code,
  const char* module,
  const char* function,
  char* msg,
  void* eh_data) 
{
  if (error_code != CV_WARNING) {
    std::cout << "From CVODE: " << msg << std::endl;
    Abort("Aborting from CVODE");
  }
}

int reactor_init(int reactor_type, int Ncells)
{
  checkCvodeOptions(reactor_type);

#ifndef AMREX_USE_GPU
  // ----------------------------------------------------------
  // On CPU, initialize cvode_mem/userData
  // ----------------------------------------------------------

  // ----------------------------------------------------------
  // Solution vector
  int neq_tot = (NUM_SPECIES + 1) * Ncells;
  y = N_VNew_Serial(neq_tot);
  if (check_flag((void*)y, "N_VNew_Serial", 0)) return (1);

  // ----------------------------------------------------------
  // Call CVodeCreate to create the solver memory and specify the Backward
  // Differentiation Formula and the use of a Newton iteration
  cvode_mem = CVodeCreate(CV_BDF);
  if (check_flag((void*)cvode_mem, "CVodeCreate", 0)) return (1);

  user_data = (CVODEUserData*)The_Arena()->alloc(sizeof(struct CVODEUserData));
  AllocUserData(user_data, reactor_type, Ncells);
  if (check_flag((void*)data, "AllocUserData", 2)) return (1);

  // Set the pointer to user-defined data
  int flag = CVodeSetUserData(cvode_mem, user_data);
  if (check_flag(&flag, "CVodeSetUserData", 1)) return (1);

  // Call CVodeInit to initialize the integrator memory and specify the user's
  // right hand side function, the inital time, and initial dependent variable
  // vector y.
  amrex::Real time = 0.0;
  flag = CVodeInit(cvode_mem, cF_RHS, time, y);
  if (check_flag(&flag, "CVodeInit", 1)) return (1);

  // ----------------------------------------------------------
  // TODO: Tolerances -> will try to harmonize with GPU, so will probably not 
  // be done here

  // ----------------------------------------------------------
  // Linear solver data
  if (user_data->isolve_type == denseFDDirect ||
      user_data->isolve_type == denseDirect ) {
    // Create dense SUNMatrix for use in linear solves
    A = SUNDenseMatrix(neq_tot, neq_tot);
    if (check_flag((void*)A, "SUNDenseMatrix", 0)) return (1);

    // Create dense SUNLinearSolver object for use by CVode
    LS = SUNDenseLinearSolver(y, A);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);

    // Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode
    flag = CVDlsSetLinearSolver(cvode_mem, LS, A);
    if (check_flag(&flag, "CVDlsSetLinearSolver", 1)) return (1);

  } else if (user_data->isolve_type == sparseDirect) {
#ifdef USE_KLU_PP
    // Create sparse SUNMatrix for use in linear solves
    A = SUNSparseMatrix(neq_tot, neq_tot, 
                        (user_data->NNZ) * user_data->ncells_d, CSC_MAT);
    if (check_flag((void*)A, "SUNSparseMatrix", 0)) return (1);

    // Create KLU solver object for use by CVode
    LS = SUNLinSol_KLU(y, A);
    if (check_flag((void*)LS, "SUNLinSol_KLU", 0)) return (1);

    // Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return (1);
#else
    Abort("sparseDirect solver_type not valid without KLU library.");
#endif

  } else if (user_data->isolve_type == customDirect) {
    // Create dense SUNMatrix for use in linear solves
    A = SUNSparseMatrix(neq_tot, neq_tot,
                        (user_data->NNZ) * user_data->ncells_d, CSR_MAT);
    if (check_flag((void*)A, "SUNDenseMatrix", 0))
      return (1);

    // Create dense SUNLinearSolver object for use by CVode
    LS = SUNLinSol_sparse_custom(y, A, reactor_type, user_data->ncells, 
                                 (NUM_SPECIES + 1), user_data->NNZ);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);

    // Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode
    flag = CVDlsSetLinearSolver(cvode_mem, LS, A);
    if (check_flag(&flag, "CVDlsSetLinearSolver", 1)) return (1);

  } else if (user_data->isolve_type == GMRES) {
    // Create the GMRES linear solver object
    LS = SUNLinSol_SPGMR(y, PREC_NONE, 0);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);

    // Set CVSpils linear solver to LS
    flag = CVSpilsSetLinearSolver(cvode_mem, LS);
    if (check_flag(&flag, "CVSpilsSetLinearSolver", 1)) return (1);

  } else if (user_data->isolve_type == precGMRES) {
    // Create the GMRES linear solver object
    LS = SUNLinSol_SPGMR(y, PREC_LEFT, 0);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);

    // Set CVSpils linear solver to LS
    flag = CVSpilsSetLinearSolver(cvode_mem, LS);
    if (check_flag(&flag, "CVSpilsSetLinearSolver", 1)) return (1);

  } else {
    Abort("Wrong choice of linear solver...");
  }
  
  // ----------------------------------------------------------
  // Analytical Jac. data for direct solver
  if (user_data->ianalytical_jacobian == 1) {
    if (user_data->isolve_type == denseDirect) {
      // Set the user-supplied Jacobian routine Jac
      flag = CVodeSetJacFn(cvode_mem, cJac);
      if (check_flag(&flag, "CVodeSetJacFn", 1)) return (1);
    } else if (user_data->isolve_type == sparseDirect) {
#ifdef USE_KLU_PP
      // Set the user-supplied KLU Jacobian routine Jac
      flag = CVodeSetJacFn(cvode_mem, cJac_KLU);
      if (check_flag(&flag, "CVodeSetJacFn", 1)) return (1);
#else
      Abort("Shouldn't be there: sparseDirect solver_type not valid without KLU library.");
#endif
    } else if (user_data->isolve_type == customDirect) {
      // Set the user-supplied Jacobian routine Jac
      flag = CVodeSetJacFn(cvode_mem, cJac_sps);
      if (check_flag(&flag, "CVodeSetJacFn", 1)) return (1);
    }
  }

  // ----------------------------------------------------------
  // Analytical Jac. data for iterative solver preconditioner
  if (user_data->iprecond_type == denseSimpleAJac) {
    // Set the JAcobian-times-vector function
    flag = CVSpilsSetJacTimes(cvode_mem, nullptr, nullptr);
    if (check_flag(&flag, "CVSpilsSetJacTimes", 1))  return (1);
    // Set the preconditioner plain dense solve and setup functions
    flag = CVSpilsSetPreconditioner(cvode_mem, Precond, PSolve);
    if (check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return (1);
  } else if (user_data->iprecond_type == sparseSimpleAJac) {
#ifdef USE_KLU_PP
    // Set the JAcobian-times-vector function
    flag = CVSpilsSetJacTimes(cvode_mem, nullptr, nullptr);
    if (check_flag(&flag, "CVSpilsSetJacTimes", 1))  return (1);
    // Set the preconditioner KLU sparse solve and setup functions
    flag = CVSpilsSetPreconditioner(cvode_mem, Precond_sparse, PSolve_sparse);
    if (check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return (1);
#else
    Abort("sparseSimpleAJac precond_type not valid without KLU library.");
#endif
  } else if (user_data->iprecond_type == customSimpleAJac) {
    // Set the JAcobian-times-vector function
    flag = CVSpilsSetJacTimes(cvode_mem, nullptr, nullptr);
    if (check_flag(&flag, "CVSpilsSetJacTimes", 1))  return (1);
    // Set the preconditioner to custom solve and setup functions
    flag = CVSpilsSetPreconditioner(cvode_mem, Precond_custom, PSolve_custom);
    if (check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return (1);
  }

  // ----------------------------------------------------------
  // CVODE runtime options
  flag = CVodeSetMaxNonlinIters(cvode_mem, 50);                     // Max newton iter.
  if (check_flag(&flag, "CVodeSetMaxNonlinIters", 1)) return (1);
  flag = CVodeSetMaxErrTestFails(cvode_mem, 100);                   // Max Err.test failure
  if (check_flag(&flag, "CVodeSetMaxErrTestFails", 1)) return (1);
  flag = CVodeSetErrHandlerFn(cvode_mem, cvodeErrHandler, 0);       // Err. handler funct.
  if (check_flag(&flag, "CVodeSetErrHandlerFn", 1)) return (1);
  flag = CVodeSetMaxNumSteps(cvode_mem, 10000);                     // Max substeps
  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return (1);
  flag = CVodeSetMaxOrd(cvode_mem, 2);                              // Max order
  if (check_flag(&flag, "CVodeSetMaxOrd", 1)) return (1);
  flag = CVodeSetJacEvalFrequency(cvode_mem, 100);                  // Max Jac age
  if (check_flag(&flag, "CVodeSetJacEvalFrequency", 1)) return (1);

  // End of CPU section
#endif

  return (0);
}

void
checkCvodeOptions(int a_reactType){

  // Query options
  ParmParse pp("ode");
  int iverbose = 1;
  pp.query("verbose", iverbose);

  if (iverbose > 0) {
    Print() << "Number of species in mech is " << NUM_SPECIES << "\n";
  }

  std::string solve_type_str = "none";
  ParmParse ppcv("cvode");
  ppcv.query("solve_type", solve_type_str);
  int isolve_type = -1;
  int ianalytical_jacobian = 0;
  int iprecond_type = 0;

#ifdef AMREX_USE_GPU
  if (solve_type_str == "sparse_direct") {
    isolve_type = sparseDirect;
    ianalytical_jacobian = 1;
#ifdef AMREX_USE_CUDA
    if (iverbose > 0) Print() << " Using a cuSparse direct linear solve with analytical Jacobian\n";
#else
    Abort("solve_type 'sparse_direct' only available with CUDA");
#endif
  } else if (solve_type_str == "custom_direct") {
    isolve_type = customDirect;
    ianalytical_jacobian = 1;
#ifdef AMREX_USE_CUDA
    if (iverbose > 0) Print() << " Using a custom direct linear solve with analytical Jacobian\n";
#else
    Abort("solve_type 'custom_direct' only available with CUDA");
#endif
  } else if (solve_type_str == "GMRES") {
    isolve_type = GMRES;
    if (iverbose > 0) Print() << " Using a JFNK GMRES linear solve\n";
  } else if (solve_type_str == "precGMRES") {
    isolve_type = precGMRES;
    if (iverbose > 0) Print() << " Using a JFNK GMRES linear solve";
    std::string prec_type_str = "cuSparse_simplified_AJacobian";
    ppcv.query("precond_type", prec_type_str);
    if (prec_type_str == "cuSparse_simplified_AJacobian") {
      iprecond_type = sparseSimpleAJac;
#ifdef AMREX_USE_CUDA
      Print() << " with a cuSparse simplified AJ-based preconditioner"; 
#else
      Abort("precond_type 'cuSparse_simplified_AJacobian' only available with CUDA");
#endif
    } else {
      Abort("Wrong precond_type. Only option is: 'cuSparse_simplified_AJacobian'");
    }
    Print() << "\n"; 
  } else {
    Abort("Wrong solve_type. Options are: 'sparse_direct', 'custom_direct', 'GMRES', 'precGMRES'");
  }

#else
  if (solve_type_str == "dense_direct") {
    isolve_type = denseFDDirect;
    if (iverbose > 0) Print() << " Using a dense direct linear solve with Finite Difference Jacobian\n";
  } else if (solve_type_str == "denseAJ_direct") {
    isolve_type = denseDirect;
    ianalytical_jacobian = 1;
    if (iverbose > 0) Print() << " Using a dense direct linear solve with Analytical Jacobian\n";
  } else if (solve_type_str == "sparse_direct") {
    isolve_type = sparseDirect;
    ianalytical_jacobian = 1;
#ifndef USE_KLU_PP
    Abort("solver_type sparse_direct requires the KLU library");
#endif
    if (iverbose > 0) Print() << " Using a sparse direct linear solve with KLU Analytical Jacobian\n";
  } else if (solve_type_str == "custom_direct") {
    isolve_type = customDirect;
    ianalytical_jacobian = 1;
    if (iverbose > 0) Print() << " Using a sparse custom direct linear solve with Analytical Jacobian\n";
  } else if (solve_type_str == "GMRES") {
    isolve_type = GMRES;
    if (iverbose > 0) Print() << " Using a JFNK GMRES linear solve\n";
  } else if (solve_type_str == "precGMRES") {
    isolve_type = precGMRES;
    if (iverbose > 0) Print() << " Using a JFNK GMRES linear solve";
    std::string prec_type_str = "none";
    ppcv.query("precond_type", prec_type_str);
    if (prec_type_str == "dense_simplified_AJacobian") {
      iprecond_type = denseSimpleAJac;
      Print() << " with a dense simplified AJ-based preconditioner"; 
    } else if (prec_type_str == "sparse_simplified_AJacobian") {
      iprecond_type = sparseSimpleAJac;
#ifndef USE_KLU_PP
      Abort("precond_type sparse_simplified_AJacobian requires the KLU library");
#endif
      Print() << " with a sparse simplified AJ-based preconditioner"; 
    } else if (prec_type_str == "custom_simplified_AJacobian") {
      iprecond_type = customSimpleAJac;
      Print() << " with a custom simplified AJ-based preconditioner"; 
    } else {
      Abort("Wrong precond_type. Options are: 'dense_simplified_AJacobian', 'sparse_simplified_AJacobian', 'custom_simplified_AJacobian'");
    }
    Print() << "\n"; 
  } else if (solve_type_str == "diagnostic") {
    isolve_type = hackDumpSparsePattern;
  } else {
    Abort("Wrong solve_type. Options are: 'dense_direct', denseAJ_direct', 'sparse_direct', 'custom_direct', 'GMRES', 'precGMRES'");
  }
#endif

  // Print additionnal information
  if (iprecond_type == sparseSimpleAJac) {
    int nJdata;
    int HP;
    if (a_reactType == eint_rho) {
      HP = 0;
    } else {
      HP = 1;
    }
    // Simplified AJ precond data
#ifdef AMREX_USE_GPU
#if defined(AMREX_USE_CUDA)
    SPARSITY_INFO_SYST_SIMPLIFIED(&nJdata, &HP);
    if (iverbose > 0) {
      Print() << "--> cuSparse AJ based matrix Preconditioner -- non zero entries: " << nJdata
              << ", which represents "
              << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
              << " % fill-in pattern\n";
    }
#elif defined(AMREX_USE_HIP)
    Abort("\n--> precond_type sparse simplified_AJacobian not available with HIP \n");
#endif

#else
    SPARSITY_INFO_SYST_SIMPLIFIED(&nJdata, &HP);
    if (iverbose > 0) {
      Print() << "--> KLU sparse AJ based matrix Preconditioner -- non zero entries: " << nJdata
              << ", which represents "
              << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
              << " % fill-in pattern\n";
    }
#endif
  } else if (iprecond_type == customSimpleAJac) {
#ifdef AMREX_USE_GPU
    Abort("Shound't be here: precond_type 'custom_simplified_AJacobian' doesn't exist on GPU");
#else
    int nJdata;
    int HP;
    if (a_reactType == eint_rho) {
      HP = 0;
    } else {
      HP = 1;
    }
    // Simplified AJ precond data
    SPARSITY_INFO_SYST_SIMPLIFIED(&nJdata, &HP);
    if (iverbose > 0) {
      Print() << "--> custom sparse AJ based matrix Preconditioner -- non zero entries: " << nJdata
              << ", which represents "
              << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
              << " % fill-in pattern\n";
    }
#endif
  }

  if (ianalytical_jacobian == 1) {
    int nJdata;
    int HP;
    int Ncells = 1;         // Print the pattern of the diagonal block. Ncells will actually vary on GPU.
    if (reactor_type == eint_rho) {
      HP = 0;
    } else {
      HP = 1;
    }
#ifdef AMREX_USE_GPU
#if defined(AMREX_USE_CUDA)
    SPARSITY_INFO_SYST(&nJdata, &HP, Ncells);
    if (iverbose > 0) {
      Print() << "--> cuSparse based matrix Solver -- non zero entries: " << nJdata
              << ", which represents "
              << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
              << " % fill-in pattern\n";
    }
#elif defined(AMREX_USE_HIP)
    Abort("\n--> Analytical Jacobian not available with HIP. Change solve_type.\n");
#endif

#else
    if (isolve_type == customDirect) {
       SPARSITY_INFO_SYST(&nJdata, &HP, Ncells);
       if (iverbose > 0) {
         Print() << "--> sparse AJ-based matrix custom Solver -- non zero entries: " << nJdata
                 << ", which represents "
                 << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
                 << " % fill-in pattern\n";
       }
    } else if ( isolve_type == sparseDirect) {
#ifdef USE_KLU_PP
       SPARSITY_INFO(&nJdata, &HP, Ncells);
       if (iverbose > 0) {
         Print() << "--> KLU sparse AJ-based matrix Solver -- non zero entries: " << nJdata
                 << ", which represents "
                 << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
                 << " % fill-in pattern\n";
       }
#else
       Abort("solver_type 'sparseDirect' uses a sparse KLU matrix and requires the KLU library");
#endif
    }
#endif
  }

  if (isolve_type == hackDumpSparsePattern) {
    // This is a diagnostic option -> dump sparsity pattern and abort.
    // Reactor type
    int HP = 1;     // defaulted to HP
    if (a_reactType == eint_rho) {
      HP = 0;
    } else {
      HP = 1;
    }

    // CHEMISTRY JAC
    int nJdata = 0;
    SPARSITY_INFO(&nJdata, &HP, 1);
    Print() << "--> Chem. Jac -- non zero entries: " << nJdata
            << ", which represents "
            << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
            << " % fill-in pattern\n";
    SUNMatrix PS;
    PS = SUNSparseMatrix( (NUM_SPECIES + 1), (NUM_SPECIES + 1), 
                          nJdata, CSR_MAT);
    int *rowCount = (int*)SUNSparseMatrix_IndexPointers(PS);
    int *colIdx = (int*)SUNSparseMatrix_IndexValues(PS);
    SPARSITY_PREPROC_CSR(colIdx, rowCount, &HP, 1, 0);
    Print() <<"\n\n *** Treating CHEM Jac (CSR symbolic analysis)*** \n\n";
    int counter = 0;
    for (int i = 0; i < NUM_SPECIES + 1; i++) {
      int nbVals = rowCount[i + 1] - rowCount[i];
      int* idx_arr = new int[nbVals];
      std::fill_n(idx_arr, nbVals, -1);
      std::memcpy(idx_arr, colIdx + rowCount[i], nbVals * sizeof(int));
      int idx = 0;
      for (int j = 0; j < NUM_SPECIES + 1; j++) {
        if ((j == idx_arr[idx]) && (nbVals > 0)) {
          std::cout << 1 << " ";
          idx = idx + 1;
          counter = counter + 1;
        } else {
          std::cout << 0 << " ";
        }
      }
      delete[] idx_arr;
      std::cout << std::endl;
    }
    Print() << " There was " << counter << " non zero elems (compared to the "
            << nJdata << " we need) \n";

    // SYST JAC
    SPARSITY_INFO_SYST(&nJdata, &HP, 1);
    Print() << "--> Syst. Jac -- non zero entries: " << nJdata
            << ", which represents "
            << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
            << " % fill-in pattern\n";
    PS = SUNSparseMatrix((NUM_SPECIES + 1), (NUM_SPECIES + 1),
                         nJdata, CSR_MAT);
    rowCount = (int*)SUNSparseMatrix_IndexPointers(PS);
    colIdx = (int*)SUNSparseMatrix_IndexValues(PS);
    SPARSITY_PREPROC_SYST_CSR(colIdx, rowCount, &HP, 1, 1);
    Print() <<"\n\n *** Treating SYST Jac (CSR symbolic analysis)*** \n\n";
    counter = 0;
    for (int i = 0; i < NUM_SPECIES + 1; i++) {
      int nbVals = rowCount[i + 1] - rowCount[i];
      int* idx_arr = new int[nbVals];
      std::fill_n(idx_arr, nbVals, -1);
      std::memcpy(idx_arr, colIdx + (rowCount[i] - 1), nbVals * sizeof(int));
      int idx = 0;
      for (int j = 0; j < NUM_SPECIES + 1; j++) {
        if ((j == idx_arr[idx] - 1) && ((nbVals - idx) > 0)) {
          std::cout << 1 << " ";
          idx = idx + 1;
          counter = counter + 1;
        } else {
          std::cout << 0 << " ";
        }
      }
      delete[] idx_arr;
      std::cout << std::endl;
    }
    Print() << " There was " << counter << " non zero elems (compared to the "
            << nJdata << " we need) \n";

    // SYST JAC SIMPLIFIED
    SPARSITY_INFO_SYST_SIMPLIFIED(&nJdata, &HP);
    Print() << "--> Simplified Syst Jac (for Precond) -- non zero entries: "
            << nJdata << ", which represents "
            << nJdata / float((NUM_SPECIES + 1) * (NUM_SPECIES + 1)) * 100.0
            << " % fill-in pattern\n";
    PS = SUNSparseMatrix((NUM_SPECIES + 1), (NUM_SPECIES + 1),
                          nJdata, CSR_MAT);
    rowCount = (int*)SUNSparseMatrix_IndexPointers(PS);
    colIdx = (int*)SUNSparseMatrix_IndexValues(PS);
    SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(colIdx, rowCount, &HP, 1);
    Print() <<"\n\n *** Treating simplified SYST Jac (CSR symbolic analysis)*** \n\n";
    counter = 0;
    for (int i = 0; i < NUM_SPECIES + 1; i++) {
      nbVals = rowCount[i + 1] - rowCount[i];
      int* idx_arr = new int[nbVals];
      std::fill_n(idx_arr, nbVals, -1);
      std::memcpy(idx_arr, colIdx + (rowCount[i] - 1), nbVals * sizeof(int));
      int idx = 0;
      for (int j = 0; j < NUM_SPECIES + 1; j++) {
        if ((j == idx_arr[idx] - 1) && ((nbVals - idx) > 0)) {
          std::cout << 1 << " ";
          idx = idx + 1;
          counter = counter + 1;
        } else {
          std::cout << 0 << " ";
        }
      }
      delete[] idx_arr;
      std::cout << std::endl;
    }
    Print() << " There was " << counter << " non zero elems (compared to the "
            << nJdata << " we need) \n";

    Abort("Chemistry sparsity pattern dumped -> exiting now !");
  }
}

int
react(
  const amrex::Box& box,
  amrex::Array4<amrex::Real> const& rY_in,
  amrex::Array4<amrex::Real> const& rY_src_in,
  amrex::Array4<amrex::Real> const& T_in,
  amrex::Array4<amrex::Real> const& rEner_in,
  amrex::Array4<amrex::Real> const& rEner_src_in,
  amrex::Array4<amrex::Real> const& FC_in,
  amrex::Array4<int> const& mask,
  amrex::Real& dt_react,
  amrex::Real& time,
  const int& reactor_type
#ifdef AMREX_USE_GPU
  ,
  amrex::gpuStream_t stream
#endif
)
{
  // Sizes
  int ncells = box.numPts();
  int neq_tot = (NUM_SPECIES + 1) * ncells;

  // CPU and GPU version are very different such that the entire file
  // is split between a GPU region and a CPU region

  //----------------------------------------------------------
  // GPU Region
  //----------------------------------------------------------
#ifdef AMREX_USE_GPU
  //----------------------------------------------------------
  // On CPU these lives as class variable and where initialized in reactor_init()
  N_Vector y = NULL;
  SUNLinearSolver LS = NULL;
  SUNMatrix A = NULL;
  void* cvode_mem = NULL;
  CVODEUserData * user_data;
  user_data = (CVODEUserData*)The_Arena()->alloc(sizeof(struct CVODEUserData));

  // Fill user_data
  Gpu::streamSynchronize();
  user_data = (CVODEUserData*)The_Arena()->alloc(sizeof(struct CVODEUserData));
  user_data->energy_init_d = (Real*)The_Device_Arena()->alloc(ncells * sizeof(Real));
  user_data->energy_ext_d  = (Real*)The_Device_Arena()->alloc(ncells * sizeof(Real));
  user_data->species_ext_d = (Real*)The_Device_Arena()->alloc(ncells * NUM_SPECIES * sizeof(Real));
  AllocUserData(user_data,reactor_type,ncells,A,stream);

  //----------------------------------------------------------
  // Solution vector and execution policy
#if defined(AMREX_USE_CUDA)
  y = N_VNewWithMemHelp_Cuda( neq_tot, /*use_managed_mem=*/true,
                             *amrex::sundials::The_SUNMemory_Helper());
  if (check_flag((void*)y, "N_VNewWithMemHelp_Cuda", 0)) return (1);
  SUNCudaExecPolicy* stream_exec_policy =
    new SUNCudaThreadDirectExecPolicy(256, stream);
  SUNCudaExecPolicy* reduce_exec_policy =
    new SUNCudaBlockReduceExecPolicy(256, 0, stream);
  N_VSetKernelExecPolicy_Cuda(y, stream_exec_policy, reduce_exec_policy);
  amrex::Real *yvec_d = N_VGetDeviceArrayPointer_Cuda(y);

#elif defined(AMREX_USE_HIP)
  y = N_VNewWithMemHelp_Hip( neq_tot, /*use_managed_mem=*/true,
                            *amrex::sundials::The_SUNMemory_Helper());
  if (check_flag((void*)y, "N_VNewWithMemHelp_Hip", 0)) return (1);
  SUNHipExecPolicy* stream_exec_policy =
    new SUNHipThreadDirectExecPolicy(256, stream);
  SUNHipExecPolicy* reduce_exec_policy =
    new SUNHipBlockReduceExecPolicy(256, 0, stream);
  N_VSetKernelExecPolicy_Hip(y, stream_exec_policy, reduce_exec_policy);
  amrex::Real *yvec_d = N_VGetDeviceArrayPointer_Hip(y);
#endif

  // Fill data
  BL_PROFILE_VAR("reactor::FlatStuff", FlatStuff);
  const auto len = amrex::length(box);
  const auto lo = amrex::lbound(box);
  amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    int icell = (k - lo.z) * len.x * len.y + (j - lo.y) * len.x + (i - lo.x);
    box_flatten(icell, NCELLS, i, j, k, 
                user_data->ireactor_type, rY_in, rY_src_in, T_in,
                rEner_in, rEner_src_in, yvec_d, user_data->species_ext_d,
                user_data->energy_ext_d, user_data->energy_init_d);
  });
  BL_PROFILE_VAR_STOP(FlatStuff);

#ifdef AMREX_USE_OMP
  Gpu::Device::streamSynchronize();
#endif

  //----------------------------------------------------------
  // Setup Cvode object
  cvode_mem = CVodeCreate(CV_BDF);
  if (check_flag((void*)cvode_mem, "CVodeCreate", 0)) return (1);
  flag = CVodeSetUserData(cvode_mem, static_cast<void*>(user_data));

  amrex::Real time_init = time;
  amrex::Real time_out = time + dt_react;

  // Call CVodeInit to initialize the integrator memory and specify the
  //  user's right hand side function, the inital time, and
  //  initial dependent variable vector y.
  flag = CVodeInit(cvode_mem, cF_RHS, time_init, y);
  if (check_flag(&flag, "CVodeInit", 1)) return (1);

  //----------------------------------------------------------
  // Setup tolerances with typical values
  atol = N_VClone(y);
#if defined(AMREX_USE_CUDA)
  ratol = N_VGetHostArrayPointer_Cuda(atol);
#elif defined(AMREX_USE_HIP)
  ratol = N_VGetHostArrayPointer_Hip(atol);
#endif
  if (typVals[0] > 0 && user_data->iverbose>1) {
    printf(
      "Setting CVODE tolerances rtol = %14.8e atolfact = %14.8e in PelePhysics "
      "\n",
      relTol, absTol);
    for (int i = 0; i < ncells; i++) {
      int offset = i * (NUM_SPECIES + 1);
      for (int k = 0; k < NUM_SPECIES + 1; k++) {
        ratol[offset + k] = typVals[k] * absTol;
      }
    }
  } else {
    for (int i = 0; i < neq_tot; i++) {
      ratol[i] = absTol;
    }
  }
#if defined(AMREX_USE_CUDA)
  N_VCopyToDevice_Cuda(atol);
#elif defined(AMREX_USE_HIP)
  N_VCopyToDevice_Hip(atol);
#endif
  // Call CVodeSVtolerances to specify the scalar relative tolerance
  // and vector absolute tolerances
  flag = CVodeSVtolerances(cvode_mem, relTol, atol);
  if (check_flag(&flag, "CVodeSVtolerances", 1)) return (1);

  // ----------------------------------------------------------
  // Linear solver data
  if (user_data->isolve_type == sparseDirect) {
#if defined(AMREX_USE_CUDA)
    LS = SUNLinSol_cuSolverSp_batchQR(y, A, user_data->cusolverHandle);
    if (check_flag((void*)LS, "SUNLinSol_cuSolverSp_batchQR", 0)) return (1);
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return (1);

#else
    Abort("Shoudn't be there. solve_type sparse_direct only available with CUDA");
#endif
  } else if (user_data->isolve_type == customDirect) {
#if defined(AMREX_USE_CUDA)
    LS = SUNLinSol_dense_custom(y, A, stream);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return (1);

    flag = CVodeSetJacFn(cvode_mem, cJac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return (1);
#else
    Abort("Shoudn't be there. solve_type custom_direct only available with CUDA");
#endif
  } else if (user_data->isolve_type == GMRES) {
    LS = SUNLinSol_SPGMR(y, PREC_NONE, 0);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);
    flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return (1);
    flag = CVodeSetJacTimes(cvode_mem, NULL, NULL);
    if (check_flag(&flag, "CVodeSetJacTimes", 1)) return (1);
  } else if (user_data->isolve_type == precGMRES) {
    LS = SUNLinSol_SPGMR(y, PREC_LEFT, 0);
    if (check_flag((void*)LS, "SUNDenseLinearSolver", 0)) return (1);
    flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return (1);
    flag = CVodeSetJacTimes(cvode_mem, NULL, NULL);
    if (check_flag(&flag, "CVodeSetJacTimes", 1)) return (1);
  }

  // ----------------------------------------------------------
  // Analytical Jac. data for direct solver
  // Both sparse/custom direct uses the same Jacobian functions
  if (user_data->ianalytical_jacobian == 1) {
    flag = CVodeSetJacFn(cvode_mem, cJac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return (1);
  }

  // ----------------------------------------------------------
  // Analytical Jac. data for iterative solver preconditioner
  if (user_data->iprecond_type == sparseSimpleAJac) {
      flag = CVodeSetPreconditioner(cvode_mem, Precond, PSolve);
      if (check_flag(&flag, "CVodeSetPreconditioner", 1)) return (1);
  }

  // ----------------------------------------------------------
  // CVODE runtime options
  flag = CVodeSetMaxNumSteps(cvode_mem, 100000);
  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return (1);
  flag = CVodeSetMaxOrd(cvode_mem, 2);
  if (check_flag(&flag, "CVodeSetMaxOrd", 1)) return (1);
  
  // ----------------------------------------------------------
  // Actual CVODE solve
  BL_PROFILE_VAR("AroundCVODE", AroundCVODE);
  flag = CVode(cvode_mem, time_out, y, &time_init, CV_NORMAL);
  if (check_flag(&flag, "CVode", 1)) return (1);
  BL_PROFILE_VAR_STOP(AroundCVODE);

#ifdef MOD_REACTOR
  dt_react = time_init - time;
  time = time_init;
#endif

#ifdef AMREX_USE_OMP
  Gpu::Device::streamSynchronize();
#endif

  // Get workload estimate
  long int nfe;
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);

  BL_PROFILE_VAR_START(FlatStuff);
  amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    int icell = (k - lo.z) * len.x * len.y + (j - lo.y) * len.x + (i - lo.x);

    box_unflatten(
      icell, NCELLS, i, j, k, user_data->ireactor_type, rY_in, T_in, rEner_in,
      rEner_src_in, FC_in, yvec_d, user_data->energy_init_d, nfe, dt_react);
  });
  BL_PROFILE_VAR_STOP(FlatStuff);

  if (user_data->iverbose > 1) {
    PrintFinalStats(cvode_mem);
  }

  //----------------------------------------------------------
  // Clean up
  N_VDestroy(y);
  CVodeFree(&cvode_mem);

  SUNLinSolFree(LS);
  if (A != nullptr) {
    SUNMatDestroy(A);
  }
  The_Device_Arena()->free(user_data->species_ext_d);
  The_Device_Arena()->free(user_data->energy_init_d);
  The_Device_Arena()->free(user_data->energy_ext_d);
  // Direct solves analytical Jacobian data
  if (user_data->isolve_type == sparseDirect) {
#ifdef AMREX_USE_CUDA
    The_Arena()->free(user_data->csr_row_count_h);
    The_Arena()->free(user_data->csr_col_index_h);
    The_Device_Arena()->free(user_data->csr_row_count_d);
    The_Device_Arena()->free(user_data->csr_col_index_d);
    cusolverStatus_t cusolver_status = cusolverSpDestroy(user_data->cusolverHandle);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cusparseStatus_t cusparse_status = cusparseDestroy(user_data->cuSPHandle);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
#endif
  } else if (user_data->isolve_type == customDirect) {
#ifdef AMREX_USE_CUDA
    The_Arena()->free(user_data->csr_row_count_h);
    The_Arena()->free(user_data->csr_col_index_h);
    The_Device_Arena()->free(user_data->csr_row_count_d);
    The_Device_Arena()->free(user_data->csr_col_index_d);
    cusparseStatus_t cusparse_status = cusparseDestroy(user_data->cuSPHandle);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
#endif
  }
  // Preconditioner analytical Jacobian data
  if (user_data->iprecond_type == sparseSimpleAJac) {
#ifdef AMREX_USE_CUDA
    The_Arena()->free(user_data->csr_row_count_h);
    The_Arena()->free(user_data->csr_col_index_h);
    The_Device_Arena()->free(user_data->csr_row_count_d);
    The_Device_Arena()->free(user_data->csr_col_index_d);
    The_Arena()->free(user_data->csr_val_h);
    The_Arena()->free(user_data->csr_jac_h);
    The_Device_Arena()->free(user_data->csr_val_d);
    The_Device_Arena()->free(user_data->csr_jac_d);

    cusolverStatus_t cusolver_status = cusolverSpDestroy(user_data->cusolverHandle);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status = cusolverSpDestroyCsrqrInfo(user_data->info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaFree(user_data->buffer_qr);
#endif
  }

  The_Arena()->free(user_data);

  N_VDestroy(atol);

  //----------------------------------------------------------
  // CPU Region
  //----------------------------------------------------------
#else

  int omp_thread = 0;
#ifdef AMREX_USE_OMP
  omp_thread = omp_get_thread_num();
#endif

  // Initial time and time to reach after integration
  time_init = time;

  // Perform integration one cell at a time
  ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (mask(i, j, k) != -1) {
      Real mass_frac[NUM_SPECIES];
      Real rho = 0.0;
      Real rho_inv;
      Real Enrg_loc;
      Real temp;

      realtype* yvec_d = N_VGetArrayPointer(y);

      BL_PROFILE_VAR("reactor::FlatStuff", FlatStuff);
      for (int n = 0; n < NUM_SPECIES; n++) {
        yvec_d[n] = rY_in(i, j, k, n);
        data->rYsrc[n] = rY_src_in(i, j, k, n);
        rho += yvec_d[n];
      }
      rho_inv = 1.0 / rho;
      temp = T_in(i, j, k, 0);
      data->rhoX_init[0] = rEner_in(i, j, k, 0);
      data->rhoXsrc_ext[0] = rEner_src_in(i, j, k, 0);

      // T update with energy and Y
      for (int n = 0; n < NUM_SPECIES; n++) {
        mass_frac[n] = yvec_d[n] * rho_inv;
      }
      Enrg_loc = data->rhoX_init[0] * rho_inv;
      auto eos = pele::physics::PhysicsType::eos();
      if (data->ireactor_type == 1) {
        eos.REY2T(rho, Enrg_loc, mass_frac, temp);
      } else {
        eos.RHY2T(rho, Enrg_loc, mass_frac, temp);
      }
      yvec_d[NUM_SPECIES] = temp;
      BL_PROFILE_VAR_STOP(FlatStuff);

      // ReInit CVODE is faster
      CVodeReInit(cvode_mem, time_init, y);

      // Time to reach after integration
      Real time_out_lcl = time_init + dt_react;

      // Integration
      Real dummy_time;
      BL_PROFILE_VAR("reactor::AroundCVODE", AroundCVODE);
      CVode(cvode_mem, time_out_lcl, y, &dummy_time, CV_NORMAL);
      BL_PROFILE_VAR_STOP(AroundCVODE);

      if ((data->iverbose > 1) && (omp_thread == 0)) {
        Print() << "Additional verbose info --\n";
        PrintFinalStats(cvode_mem);
        Print() << "\n -------------------------------------\n";
      }

      // Get estimate of how hard the integration process was
      long int nfe = 0;
      long int nfeLS = 0;
      CVodeGetNumRhsEvals(cvode_mem, &nfe);
      CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
      FC_in(i, j, k, 0) = nfe + nfeLS;

      BL_PROFILE_VAR_START(FlatStuff);
      rho = 0.0;
      for (int n = 0; n < NUM_SPECIES; n++) {
        rY_in(i, j, k, n) = yvec_d[n];
        rho += yvec_d[n];
      }
      rho_inv = 1.0 / rho;
      temp = yvec_d[NUM_SPECIES];

      // T update with energy and Y
      for (int n = 0; n < NUM_SPECIES; n++) {
        mass_frac[n] = yvec_d[n] * rho_inv;
      }
      rEner_in(i, j, k, 0) =
        data->rhoX_init[0] + (dummy_time - time_init) * data->rhoXsrc_ext[0];
      Enrg_loc = rEner_in(i, j, k, 0) * rho_inv;
      if (data->ireactor_type == 1) {
        eos.REY2T(rho, Enrg_loc, mass_frac, temp);
      } else {
        eos.RHY2T(rho, Enrg_loc, mass_frac, temp);
      }
      T_in(i, j, k, 0) = temp;
      BL_PROFILE_VAR_STOP(FlatStuff);

      if ((data->iverbose > 3) && (omp_thread == 0)) {
        Print() << "END : time curr is " << dummy_time
                << " and actual dt_react is " << (dummy_time - time_init)
                << "\n";
      }
    } else {
      FC_in(i, j, k, 0) = 0.0;
    }
  });

  // Update dt_react with real time step taken ...
  // should be very similar to input dt_react
  // dt_react = dummy_time - time_init;
#ifdef MOD_REACTOR
  // If reactor mode is activated, update time to perform subcycling
  time = time_init + dt_react;
#endif

  long int nfe = 20; // Dummy, the return value is no longer used for this function.
#endif

  return nfe;
}

// React for 1d arrays
int
react(
  realtype *rY_in,
  realtype *rY_src_in, 
  realtype *rX_in,
  realtype *rX_src_in,
  realtype &dt_react,
  realtype &time,
  int reactor_type,
  int Ncells
#ifdef AMREX_USE_GPU
  ,
  amrex::gpuStream_t stream
#endif
)
{
  //TODO
  Abort("1dArray version is TODO");
}

void
AllocUserData(CVODEUserData* udata,
              int a_reactType,
              int a_ncells
#ifdef AMREX_USE_GPU
              , SUNMatrix a_A,
              amrex::gpuStream_t stream
#endif
)
{
  //----------------------------------------------------------
  // Query options
  ParmParse pp("ode");
  int iverbose = 1;
  pp.query("verbose", iverbose);

  std::string solve_type_str = "none";
  ParmParse ppcv("cvode");
  ppcv.query("solve_type", solve_type_str);

#ifdef AMREX_USE_GPU
  if (solve_type_str == "sparse_direct") {
    udata->isolve_type = sparseDirect;
    udata->ianalytical_jacobian = 1;
  } else if (solve_type_str == "custom_direct") {
    udata->isolve_type = customDirect;
    udata->ianalytical_jacobian = 1;
  } else if (solve_type_str == "GMRES") {
    udata->isolve_type = GMRES;
  } else if (solve_type_str == "precGMRES") {
    udata->isolve_type = precGMRES;
    std::string prec_type_str = "cuSparse_simplified_AJacobian";
    ppcv.query("precond_type", prec_type_str);
    if (prec_type_str == "cuSparse_simplified_AJacobian") {
       udata->iprecond_type = sparseSimpleAJac;
    } else {
      Abort("Wrong precond_type. Only option is: 'cuSparse_simplified_AJacobian'");
    }
    Print() << "\n"; 
  } else {
    Abort("Wrong solve_type. Options are: 'sparse_direct', 'custom_direct', 'GMRES', 'precGMRES'");
  }

#else
  if (solve_type_str == "dense_direct") {
    udata->isolve_type = denseFDDirect;
  } else if (solve_type_str == "denseAJ_direct") {
    udata->isolve_type = denseDirect;
    ianalytical_jacobian = 1;
  } else if (solve_type_str == "sparse_direct") {
    udata->isolve_type = sparseDirect;
    ianalytical_jacobian = 1;
#ifndef USE_KLU_PP
    Abort("solver_type sparse_direct requires the KLU library");
#endif
  } else if (solve_type_str == "custom_direct") {
    udata->isolve_type = customDirect;
    ianalytical_jacobian = 1;
  } else if (solve_type_str == "GMRES") {
    udata->isolve_type = GMRES;
  } else if (solve_type_str == "precGMRES") {
    udata->isolve_type = precGMRES;
    std::string prec_type_str = "sparse_simplified_AJacobian";
    ppcv.query("precond_type", prec_type_str);
    if (prec_type_str == "dense_simplified_AJacobian") {
      udata->iprecond_type = denseSimpleAJac;
    } else if (prec_type_str == "sparse_simplified_AJacobian") {
      udata->iprecond_type = sparseSimpleAJac;
#ifndef USE_KLU_PP
      Abort("precond_type sparse_simplified_AJacobian requires the KLU library");
#endif
    } else if (prec_type_str == "custom_simplified_AJacobian") {
      udata->iprecond_type = customSimpleAJac;
    } else {
      Abort("Wrong precond_type. Options are: 'dense_simplified_AJacobian', 'sparse_simplified_AJacobian', 'custom_simplified_AJacobian'");
    }
    Print() << "\n"; 
  } else {
    Abort("Wrong solve_type. Options are: 'dense_direct', denseAJ_direct', 'sparse_direct', 'custom_direct', 'GMRES', 'precGMRES'");
  }
#endif

  //----------------------------------------------------------
  // Pass options to udata
  int HP = (a_reactType==1) ? 0 : 1;
  int neq_tot   = (NUM_SPECIES + 1) * a_ncells;
  int nspec_tot = (NUM_SPECIES) * a_ncells;
  udata->ireactor_type = a_reactType;
  udata->ncells_d      = a_ncells;
  udata->iverbose    = iverbose;
#ifdef AMREX_USE_GPU
  udata->nbThreads   = 32;
  udata->nbBlocks    = std::max(1, a_ncells / user_data->nbThreads);
  udata->stream      = stream;
#endif

  //----------------------------------------------------------
  // Alloc internal udata solution/forcing containers
  usdata->species_ext_d = (Real*)The_Device_Arena()->alloc(nspec_tot * sizeof(Real));
  usdata->energy_init_d = (Real*)The_Device_Arena()->alloc(a_ncells * sizeof(Real));
  usdata->energy_ext_d  = (Real*)The_Device_Arena()->alloc(a_ncells * sizeof(Real));

#ifndef AMREX_USE_GPU
  // TODO: some of these should exist for GPU. Not sure where they are right now.
  udata->yvect_sol   = (Real*)The_Device_Arena()->alloc(neq_tot * sizeof(Real));
  udata->FCunt       = (int*)The_Device_Arena()->alloc(a_ncells * sizeof(int));
  udata->mask        = (int*)The_Device_Arena()->alloc(a_ncells * sizeof(int));
  udata->FirstTimePrecond = true;
  udata->reactor_cvode_initialized = false;
  udata->actual_ok_to_react = true;
#endif

  //----------------------------------------------------------
  // Alloc internal udata Analytical Jacobian containers
#ifdef AMREX_USE_GPU
  if (udata->isolve_type == sparseDirect) {
#ifdef AMREX_USE_CUDA
    SPARSITY_INFO_SYST(&(udata->NNZ),&HP,1);
    udata->csr_row_count_h = (int*)The_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
    udata->csr_col_index_h = (int*)The_Arena()->alloc(udata->NNZ * sizeof(int));
    udata->csr_row_count_d = (int*) The_Device_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
    udata->csr_col_index_d = (int*) The_Device_Arena()->alloc(udata->NNZ * sizeof(int));

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusolver_status = cusolverSpSetStream(udata->cusolverHandle, stream);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cusolver_status = cusolverSpSetStream(udata->cusolverHandle, stream);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
  
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparse_status = cusparseCreate(&(udata->cuSPHandle));
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusparse_status = cusparseSetStream(udata->cuSPHandle, stream);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    
    a_A = SUNMatrix_cuSparse_NewBlockCSR(a_ncells, (NUM_SPECIES + 1), (NUM_SPECIES + 1),
                                         udata->NNZ, udata->cuSPHandle); 
    if (check_flag((void *)a_A, "SUNMatrix_cuSparse_NewBlockCSR", 0)) return(1);

    int retval = SUNMatrix_cuSparse_SetFixedPattern(a_A, 1);
    if (check_flag(&retval, "SUNMatrix_cuSparse_SetFixedPattern", 1)) return(1);

    SPARSITY_PREPROC_SYST_CSR(user_data->csr_col_index_h, user_data->csr_row_count_h, &HP, 1, 0);
    amrex::Gpu::htod_memcpy(&udata->csr_col_index_d,&udata->csr_col_index_h,
                            sizeof(udata->csr_col_index_h));
    amrex::Gpu::htod_memcpy(&udata->csr_row_count_d,&udata->csr_row_count_h,
                            sizeof(udata->csr_row_count_h));
    SUNMatrix_cuSparse_CopyToDevice(a_A, NULL, udata->csr_row_count_h, udata->csr_col_index_h);
#else
    Abort("Solver_type sparse_direct is only available with CUDA on GPU");
#endif
  } else if (udata->isolve_type == customDirect) {
#ifdef AMREX_USE_CUDA
    SPARSITY_INFO_SYST(&(udata->NNZ),&HP,1);
    udata->csr_row_count_h = (int*) The_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
    udata->csr_col_index_h = (int*) The_Arena()->alloc(udata->NNZ * sizeof(int));
    udata->csr_row_count_d = (int*) The_Device_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
    udata->csr_col_index_d = (int*) The_Device_Arena()->alloc(udata->NNZ * sizeof(int));

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparse_status = cusparseCreate(&(udata->cuSPHandle));
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusparse_status = cusparseSetStream(udata->cuSPHandle, stream);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    a_A = SUNMatrix_cuSparse_NewBlockCSR(a_ncells, (NUM_SPECIES + 1), (NUM_SPECIES + 1), 
                                       udata->NNZ, udata->cuSPHandle);
    if (check_flag((void *)a_A, "SUNMatrix_cuSparse_NewBlockCSR", 0)) return(1);

    int retval = SUNMatrix_cuSparse_SetFixedPattern(a_A, 1); 
    if(check_flag(&retval, "SUNMatrix_cuSparse_SetFixedPattern", 1)) return(1);

    BL_PROFILE_VAR_START(SparsityStuff);
    SPARSITY_PREPROC_SYST_CSR(udata->csr_col_index_h, udata->csr_row_count_h, &HP, 1, 0); 
    amrex::Gpu::htod_memcpy(&udata->csr_col_index_d,&udata->csr_col_index_h,
                            sizeof(udata->csr_col_index_h));
    amrex::Gpu::htod_memcpy(&udata->csr_row_count_d,&udata->csr_row_count_h,
                            sizeof(udata->csr_row_count_h));
    SUNMatrix_cuSparse_CopyToDevice(a_A, NULL, udata->csr_row_count_h, udata->csr_col_index_h);
    BL_PROFILE_VAR_STOP(SparsityStuff);
#else
    Abort("Solver_type custom_direct is only available with CUDA on GPU");
#endif
  }

#else
  if (udata->isolve_type == sparseDirect) {
#ifdef USE_KLU_PP
    // CSC matrices data -> one big matrix used for the direct solve
    udata->colPtrs = new int*[1];
    udata->rowVals = new int*[1];
    udata->Jdata   = new amrex::Real*[1];

    // Number of non zero elements in ODE system
    SPARSITY_INFO(&(udata->NNZ), &HP, udata->ncells_d);
    // Build Sparse Matrix for direct sparse KLU solver
    (udata->PS) = new SUNMatrix[1];
    (udata->PS)[0] = SUNSparseMatrix((NUM_SPECIES + 1) * udata->ncells_d, (NUM_SPECIES + 1) * udata->ncells_d,
                                     udata->NNZ*udata->ncells_d, CSC_MAT);
    udata->colPtrs[0] = (int*)SUNSparseMatrix_IndexPointers((udata->PS)[0]);
    udata->rowVals[0] = (int*)SUNSparseMatrix_IndexValues((udata->PS)[0]);
    udata->Jdata[0]   = SUNSparseMatrix_Data((udata->PS)[0]);
    SPARSITY_PREPROC_CSC(udata->rowVals[0], udata->colPtrs[0], &HP, udata->ncells_d);
#endif
  } else if (udata->isolve_type == customDirect) {
    // Number of non zero elements in ODE system
    SPARSITY_INFO_SYST(&(udata->NNZ), &HP, udata->ncells_d);
    // Build the SUNmatrix as CSR sparse and fill ptrs to row/Vals
    udata->PSc = SUNSparseMatrix((NUM_SPECIES + 1) * udata->ncells_d, (NUM_SPECIES + 1) * udata->ncells_d,
                                  udata->NNZ * udata->ncells_d, CSR_MAT);
    udata->rowPtrs_c = (int*)SUNSparseMatrix_IndexPointers(udata->PSc);
    udata->colVals_c = (int*)SUNSparseMatrix_IndexValues(udata->PSc);
    SPARSITY_PREPROC_SYST_CSR(udata->colVals_c, udata->rowPtrs_c, &HP, udata->ncells_d, 0);
  }
#endif

  //----------------------------------------------------------
  // Alloc internal udata Preconditioner containers
#ifdef AMREX_USE_GPU
  if (udata->iprecond_type == sparseSimpleAJac) {
#ifdef AMREX_USE_CUDA
    SPARSITY_INFO_SYST_SIMPLIFIED(&(udata->NNZ),&HP); 
    udata->csr_row_count_h = (int*) The_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
    udata->csr_col_index_h = (int*) The_Arena()->alloc(udata->NNZ * sizeof(int));
    udata->csr_jac_h       = (amrex::Real*) The_Arena()->alloc(udata->NNZ * a_ncells * sizeof(amrex::Real));
    udata->csr_val_h       = (amrex::Real*) The_Arena()->alloc(udata->NNZ * a_ncells * sizeof(amrex::Real));

    udata->csr_row_count_d = (int*) The_Device_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
    udata->csr_col_index_d = (int*) The_Device_Arena()->alloc(udata->NNZ * sizeof(int));
    udata->csr_jac_d       = (amrex::Real*) The_Device_Arena()->alloc(udata->NNZ * a_ncells * sizeof(amrex::Real));
    udata->csr_val_d       = (amrex::Real*) The_Device_Arena()->alloc(udata->NNZ * a_ncells * sizeof(amrex::Real));

    SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(udata->csr_col_index_h, udata->csr_row_count_h, &HP,1);

    amrex::Gpu::htod_memcpy(&udata->csr_col_index_d,&udata->csr_col_index_h,sizeof(udata->NNZ*sizeof(int)));
    amrex::Gpu::htod_memcpy(&udata->csr_row_count_d,&udata->csr_row_count_h,sizeof((NUM_SPECIES+2)*sizeof(int)));

    size_t workspaceInBytes = 0;
    size_t internalDataInBytes = 0;

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusolver_status = cusolverSpCreate(&(udata->cusolverHandle));
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cusolver_status = cusolverSpSetStream(udata->cusolverHandle, stream);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
            
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparse_status = cusparseCreateMatDescr(&(udata->descrA)); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusparse_status = cusparseSetMatType(udata->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusparse_status = cusparseSetMatIndexBase(udata->descrA, CUSPARSE_INDEX_BASE_ONE);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusolver_status = cusolverSpCreateCsrqrInfo(&(udata->info));
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // symbolic analysis
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(udata->cusolverHandle,
                                                      NUM_SPECIES+1, // size per subsystem
                                                      NUM_SPECIES+1, // size per subsystem
                                                      udata->NNZ,
                                                      udata->descrA,
                                                      udata->csr_row_count_h,
                                                      udata->csr_col_index_h,
                                                      udata->info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    /*
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaStat1 = cudaMemGetInfo( &free_mem, &total_mem );
    assert( cudaSuccess == cudaStat1 );
    std::cout<<"(AFTER SA) Free: "<< free_mem<< " Tot: "<<total_mem<<std::endl;
    */

    // allocate working space 
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(udata->cusolverHandle,
                                                        NUM_SPECIES+1, // size per subsystem
                                                        NUM_SPECIES+1, // size per subsystem
                                                        udata->NNZ,
                                                        udata->descrA,
                                                        udata->csr_val_h,
                                                        udata->csr_row_count_h,
                                                        udata->csr_col_index_h,
                                                        a_ncells,
                                                        udata->info,
                                                        &internalDataInBytes,
                                                        &workspaceInBytes);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaError_t cudaStat1 = cudaSuccess;
    cudaStat1 = cudaMalloc((void**)&(udata->buffer_qr), workspaceInBytes);
    assert(cudaStat1 == cudaSuccess);
#else
    Abort("cuSparse_simplified_AJacobian is only available with CUDA on GPU");
#endif
  }

#else
  if (udata->iprecond_type == denseSimpleAJac) {
    // Matrix data : big bunch of dimensions, not sure why. Generally ncells == 1 so not too bad
    // Simply create the space.
    (udata->P) = new amrex::Real***[udata->ncells_d];
    (udata->Jbd) = new amrex::Real***[udata->ncells_d];
    (udata->pivot) = new sunindextype**[udata->ncells_d];
    for (int i = 0; i < udata->ncells; ++i) {
      (udata->P)[i] = new amrex::Real**[udata->ncells_d];
      (udata->Jbd)[i] = new amrex::Real**[udata->ncells_d];
      (udata->pivot)[i] = new sunindextype*[udata->ncells_d];
    }
    for (int i = 0; i < udata->ncells_d; ++i) {
      (udata->P)[i][i] = newDenseMat(NUM_SPECIES + 1, NUM_SPECIES + 1);
      (udata->Jbd)[i][i] = newDenseMat(NUM_SPECIES + 1, NUM_SPECIES + 1);
      (udata->pivot)[i][i] = newIndexArray(NUM_SPECIES + 1);
    }
  } else if ( udata->iprecond_type == sparseSimpleAJac) {
#ifdef USE_KLU_PP
    // CSC matrices data for each submatrix (cells)
    udata->colPtrs = new int*[udata->ncells_d];
    udata->rowVals = new int*[udata->ncells_d];
    udata->Jdata   = new amrex::Real*[udata->ncells_d];

    // KLU internal storage
    udata->Common = new klu_common[udata->ncells_d];
    udata->Symbolic = new klu_symbolic*[udata->ncells_d];
    udata->Numeric = new klu_numeric*[udata->ncells_d];
    // Sparse Matrices for It Sparse KLU block-solve
    udata->PS = new SUNMatrix[udata->ncells_d];
    // Number of non zero elements
    SPARSITY_INFO_SYST_SIMPLIFIED(&(udata->NNZ), &HP);
    // Not used yet. TODO use to fetch sparse Mat
    udata->indx = new int[udata->NNZ];
    udata->JSPSmat = new amrex::Real*[udata->ncells_d];
    for (int i = 0; i < udata->ncells_d; ++i) {
      (udata->PS)[i] = SUNSparseMatrix(NUM_SPECIES + 1, NUM_SPECIES + 1, udata->NNZ, CSC_MAT);
      udata->colPtrs[i] = (int*)SUNSparseMatrix_IndexPointers((udata->PS)[i]);
      udata->rowVals[i] = (int*)SUNSparseMatrix_IndexValues((udata->PS)[i]);
      udata->Jdata[i] = SUNSparseMatrix_Data((udata->PS)[i]);
      // indx not used YET
      SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
        udata->rowVals[i], udata->colPtrs[i], udata->indx, &HP);
      udata->JSPSmat[i] = new amrex::Real[(NUM_SPECIES + 1) * (NUM_SPECIES + 1)];
      klu_defaults(&(udata->Common[i]));
      // udata->Common.btf = 0;
      //(udata->Common[i]).maxwork = 15;
      // udata->Common.ordering = 1;
      udata->Symbolic[i] = klu_analyze(
        NUM_SPECIES + 1, udata->colPtrs[i], udata->rowVals[i],
        &(udata->Common[i]));
    }
#endif
  } else if ( udata->iprecond_type == customSimpleAJac) {
    // CSR matrices data for each submatrix (cells)
    udata->colVals = new int*[udata->ncells_d];
    udata->rowPtrs = new int*[udata->ncells_d];
    // Matrices for each sparse custom block-solve
    udata->PS = new SUNMatrix[udata->ncells_d];
    udata->JSPSmat = new amrex::Real*[udata->ncells_d];
    // Number of non zero elements
    SPARSITY_INFO_SYST_SIMPLIFIED(&(udata->NNZ), &HP);
    for (int i = 0; i < udata->ncells; ++i) {
      (udata->PS)[i] = SUNSparseMatrix(NUM_SPECIES + 1, NUM_SPECIES + 1, udata->NNZ, CSR_MAT);
      udata->rowPtrs[i] = (int*)SUNSparseMatrix_IndexPointers((udata->PS)[i]);
      udata->colVals[i] = (int*)SUNSparseMatrix_IndexValues((udata->PS)[i]);
      udata->Jdata[i]   = SUNSparseMatrix_Data((udata->PS)[i]);
      SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(udata->colVals[i], udata->rowPtrs[i], &HP, 0);
      udata->JSPSmat[i] = new amrex::Real[(NUM_SPECIES + 1) * (NUM_SPECIES + 1)];
    }
  }
#endif

}

static int
cF_RHS(Real t,
       N_Vector y_in,
       N_Vector ydot_in,
       void* user_data)
{
  BL_PROFILE("Pele::cF_RHS()");
#if defined(AMREX_USE_CUDA)
  realtype* yvec_d = N_VGetDeviceArrayPointer_Cuda(y_in);
  realtype* ydot_d = N_VGetDeviceArrayPointer_Cuda(ydot_in);
#elif defined(AMREX_USE_HIP)
  realtype* yvec_d = N_VGetDeviceArrayPointer_Hip(y_in);
  realtype* ydot_d = N_VGetDeviceArrayPointer_Hip(ydot_in);
#else
  realtype* yvec_d = N_VGetArrayPointer(y_in);
  realtype* ydot_d = N_VGetArrayPointer(ydot_in);
#endif

  CVODEUserData * udata = static_cast<CVODEUserData*>(user_data);
  udata->dt_save = t;

  auto ncells = udata->ncells_d;
  auto dt_save = udata->dt_save;
  auto reactor_type = udata->ireactor_type;
  auto energy_init  = udata->energy_init_d;
  auto energy_ext   = udata->energy_ext_d;
  auto species_ext  = udata->species_ext_d;
  amrex::ParallelFor(ncells,, [=] AMREX_GPU_DEVICE(int icell) noexcept {
    fKernelSpec(icell, ncells, dt_save, reactor_type, yvec_d, ydot_d,
                energy_init, energy_ext, species_ext);
  });
  Gpu::Device::streamSynchronize();

  return (0);
}

void
check_state(N_Vector yvec)
{
  /* TODO
  Real* ydata = N_VGetArrayPointer(yvec);

  data->actual_ok_to_react = true;

  for (int tid = 0; tid < data->ncells; tid++) {
    // Offset in case several cells
    int offset = tid * (NUM_SPECIES + 1);
    // rho MKS
    realtype rho = 0.0;
    for (int k = 0; k < NUM_SPECIES; k++) {
      rho = rho + ydata[offset + k];
    }
    realtype Temp = ydata[offset + NUM_SPECIES];
    if ((rho < 1.0e-10) || (rho > 1.e10)) {
      data->actual_ok_to_react = false;
      Print() << "rho " << rho << "\n";
    }
    if ((Temp < 200.0) || (Temp > 5000.0)) {
      data->actual_ok_to_react = false;
      Print() << "Temp " << Temp << "\n";
    }
  }
  */
}

void
reactor_close()
{
#ifndef AMREX_USE_GPU
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);

  if (data->isolve_type == dense_solve) {
    SUNMatDestroy(A);
  }

  N_VDestroy(y);
  FreeUserData(data);
#endif
}

// Free data memory
// Probably not complete, how about the stuff allocated in KLU mode ?
void
FreeUserData(CVODEUserData *data_wk)
{
#ifndef AMREX_USE_GPU
  delete[](data_wk->Yvect_full);
  delete[](data_wk->species_ext_d);
  delete[](data_wk->energy_init_d);
  delete[](data_wk->energy_ext_d);
  delete[](data_wk->FCunt);
  delete[](data_wk->mask);

  delete[]data_wk->colPtrs;
  delete[]data_wk->rowVals;
  delete[]data_wk->Jdata;
#ifndef USE_KLU_PP
  if (data_wk->isolve_type == iterative_gmres_solve) {
    for (int i = 0; i < data_wk->ncells; ++i) {
      destroyMat((data_wk->P)[i][i]);
      destroyMat((data_wk->Jbd)[i][i]);
      destroyArray((data_wk->pivot)[i][i]);
    }
    for (int i = 0; i < data_wk->ncells; ++i) {
      delete[](data_wk->P)[i];
      delete[](data_wk->Jbd)[i];
      delete[](data_wk->pivot)[i];
    }
    delete[](data_wk->P);
    delete[](data_wk->Jbd);
    delete[](data_wk->pivot);
    //}

#else
  if (data_wk->isolve_type == sparse_solve) {
    SUNMatDestroy(A);
    SUNMatDestroy((data_wk->PS)[0]);
    delete[](data_wk->PS);
  } else if (data_wk->isolve_type == iterative_gmres_solve) {
    delete[] data_wk->indx;
    for (int i = 0; i < data_wk->ncells; ++i) {
      klu_free_symbolic(&(data_wk->Symbolic[i]), &(data_wk->Common[i]));
      klu_free_numeric(&(data_wk->Numeric[i]), &(data_wk->Common[i]));
      delete[] data_wk->JSPSmat[i];
      SUNMatDestroy((data_wk->PS)[i]);
    }
    delete[] data_wk->JSPSmat;
    delete[] data_wk->Common;
    delete[] data_wk->Symbolic;
    delete[] data_wk->Numeric;
    delete[] data_wk->PS;
    //}
#endif

  } else if (data_wk->isolve_type == iterative_gmres_solve_custom) {
    for (int i = 0; i < data_wk->ncells; ++i) {
      delete[] data_wk->JSPSmat[i];
      SUNMatDestroy((data_wk->PS)[i]);
    }
    delete[] data_wk->colVals;
    delete[] data_wk->rowPtrs;
    delete[] data_wk->PS;
    delete[] data_wk->JSPSmat;
  } else if (data_wk->isolve_type == sparse_custom_solve) {
    SUNMatDestroy(A);
    SUNMatDestroy(data_wk->PSc);
  } else if (data_wk->isolve_type == hack_dump_sparsity_pattern) {
  }

  free(data_wk);
#endif
}

static void
PrintFinalStats(void* cvodeMem)
{
  long lenrw, leniw;
  long lenrwLS, leniwLS;
  long int nst, nfe, nsetups, nni, ncfn, netf;
  long int nli, npe, nps, ncfl, nfeLS;
  int flag;

  flag = CVodeGetWorkSpace(cvodeMem, &lenrw, &leniw);
  check_flag(&flag, "CVodeGetWorkSpace", 1);
  flag = CVodeGetNumSteps(cvodeMem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvodeMem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvodeMem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvodeMem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvodeMem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvodeMem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  flag = CVodeGetLinWorkSpace(cvodeMem, &lenrwLS, &leniwLS);
  check_flag(&flag, "CVodeGetLinWorkSpace", 1);
  flag = CVodeGetNumLinIters(cvodeMem, &nli);
  check_flag(&flag, "CVodeGetNumLinIters", 1);
  // flag = CVodeGetNumJacEvals(cvodeMem, &nje);
  // check_flag(&flag, "CVodeGetNumJacEvals", 1);
  flag = CVodeGetNumLinRhsEvals(cvodeMem, &nfeLS);
  check_flag(&flag, "CVodeGetNumLinRhsEvals", 1);

  flag = CVodeGetNumPrecEvals(cvodeMem, &npe);
  check_flag(&flag, "CVodeGetNumPrecEvals", 1);
  flag = CVodeGetNumPrecSolves(cvodeMem, &nps);
  check_flag(&flag, "CVodeGetNumPrecSolves", 1);

  flag = CVodeGetNumLinConvFails(cvodeMem, &ncfl);
  check_flag(&flag, "CVodeGetNumLinConvFails", 1);

#ifdef AMREX_USE_OMP
  Print() << "\nFinal Statistics: "
          << "(thread:" << omp_get_thread_num() << ", ";
  Print() << "cvodeMem:" << cvodeMem << ")\n";
#else
  Print() << "\nFinal Statistics:\n";
#endif
  Print() << "lenrw      = " << lenrw << "    leniw         = " << leniw
          << "\n";
  Print() << "lenrwLS    = " << lenrwLS << "    leniwLS       = " << leniwLS
          << "\n";
  Print() << "nSteps     = " << nst << "\n";
  Print() << "nRHSeval   = " << nfe << "    nLinRHSeval   = " << nfeLS << "\n";
  Print() << "nnLinIt    = " << nni << "    nLinIt        = " << nli << "\n";
  Print() << "nLinsetups = " << nsetups << "    nErrtf        = " << netf
          << "\n";
  Print() << "nPreceval  = " << npe << "    nPrecsolve    = " << nps << "\n";
  Print() << "nConvfail  = " << ncfn << "    nLinConvfail  = " << ncfl
          << "\n\n";
}
