# AMReX
DIM        = 3
PRECISION  = DOUBLE
PROFILE    = FALSE
VERBOSE    = FALSE
DEBUG      = FALSE

# Compiler
COMP	     = gnu
USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = FALSE
USE_HIP    = FALSE

# PelePhysics
FUEGO_GAS  = TRUE
TINY_PROFILE = FALSE

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    ?= ../../..

#######################
# this flag activates the subcycling mode in the D/Cvode routines
DEFINES  += -DMOD_REACTOR

#######################
# ODE solver OPTIONS: DVODE (default) / SUNDIALS / RK explicit
#######################
# Activates use of SUNDIALS: CVODE (default) / ARKODE

#######################
Eos_dir         = Fuego
Chemistry_Model = drm19
Transport_dir   = Simple
Reactor_dir = arkode

Bpack   := ./Make.package
Blocs   := .
USE_ARKODE_PP=TRUE
DEFINES += -DCVODE_BOXINTEG
include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
