/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <mpi.h>
#include <cstring>
#include "compute_temp_dumbbellrot.h"
#include "atom.h"
#include "neighbor.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "comm.h"
#include "group.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempDumbbellRot::ComputeTempDumbbellRot(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute temp command");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;

  vector = new double[6];
}

/* ---------------------------------------------------------------------- */

ComputeTempDumbbellRot::~ComputeTempDumbbellRot()
{
  if (!copymode)
    delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeTempDumbbellRot::setup()
{
  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}

/* ---------------------------------------------------------------------- */

void ComputeTempDumbbellRot::dof_compute()
{
  natoms_temp = group->count(igroup)/2;   // Number of molecules
  dof = domain->dimension * natoms_temp;  // Spatial dimensions (2)
  tfactor = force->mvv2e / (dof * force->boltz);
}

/* ---------------------------------------------------------------------- */

double ComputeTempDumbbellRot::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  int i1, i2;
  double **x = atom->x;
  double **v = atom->v;
  double *mass = atom->mass;
  double vcom [3];  // Center of mass velocity
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  double t = 0.0;

  for (int n = 0; n < nbondlist; n++) {
    if (x[bondlist[n][0]][1] < x[bondlist[n][1]][1]) { // Set i1 as left atom
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
    }
    else {
      i2 = bondlist[n][0];
      i1 = bondlist[n][1];
    }
    if (i1 < nlocal){  // Only count this dumbbell if first atom is non-ghost
      for (int j=0; j < 3; j++)
        vcom[j] = 0.5*(v[i1][j] - v[i2][j]);
      t += (vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2]) * 2 * mass[type[i1]];
    }
  }
  MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  scalar *= tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempDumbbellRot::compute_vector()
{
  invoked_vector = update->ntimestep;
  int i, i1, i2;
  double **v = atom->v;
  double *mass = atom->mass;
  double vcom [3];  // Center of mass velocity
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  double t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  for (int n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    if (i1 < nlocal){
      for (int j=0; j < 3; j++)
        vcom[j] = 0.5*(v[i1][j] + v[i2][j]);
      t[0] += mass[type[i]] * vcom[0]*vcom[0];
      t[1] += mass[type[i]] * vcom[1]*vcom[1];
      t[2] += mass[type[i]] * vcom[2]*vcom[2];
      t[3] += mass[type[i]] * vcom[0]*vcom[1];
      t[4] += mass[type[i]] * vcom[0]*vcom[2];
      t[5] += mass[type[i]] * vcom[1]*vcom[2];
    }
  }

  MPI_Allreduce(t,vector,6,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}
