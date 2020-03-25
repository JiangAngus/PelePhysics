
CHEMINP=chem.inp
THERMINP=therm.dat
FINALFILE=jp8-lu-skl.cpp

FMC=${PELE_PHYSICS_HOME}/Support/Fuego/Pythia/products/bin/fmc.py
HEADERDIR=${PELE_PHYSICS_HOME}/Support/Fuego/Mechanism/Models/header

${FUEGO_PYTHON} ${FMC} -mechanism=${CHEMINP} -thermo=${THERMINP} -name=${FINALFILE}

echo Compiling ${FINALFILE}...
