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

#include <cstdlib>
#include <cstring>
#include <cmath>
#include "compute_stress_com_atom.h"
#include "neighbor.h"
#include "atom.h"
#include "update.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
enum{NOBIAS,BIAS};

/* ---------------------------------------------------------------------- */

ComputeStressCOMAtom::ComputeStressCOMAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  id_temp(NULL), stress(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal compute stress/com/atom command");
  if (strcmp(arg[3],"NULL") == 0) id_temp = NULL;
  else {
    int n = strlen(arg[3]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[3]);
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute stress/com/atom temperature ID");
    if (modify->compute[icompute]->tempflag == 0)
      error->all(FLERR,
                 "Compute stress/com/atom temperature ID does not "
                 "compute temperature");
  }


  peratom_flag = 1;
  size_peratom_cols = 4;
  pressatomflag = 1;
  timeflag = 1;
  comm_reverse = 4;
  nmax = 0;
  keflag = 1;
 }

/* ---------------------------------------------------------------------- */

ComputeStressCOMAtom::~ComputeStressCOMAtom()
{
  delete [] id_temp;
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

void ComputeStressCOMAtom::init()
{
  // set temperature compute, must be done in init()
  // fixes could have changed or compute_modify could have changed it

  if (id_temp) {
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute stress/atom temperature ID");
    temperature = modify->compute[icompute];
    if (temperature->tempbias) biasflag = BIAS;
    else biasflag = NOBIAS;
  } else biasflag = NOBIAS;
}

/* ---------------------------------------------------------------------- */

void ComputeStressCOMAtom::compute_peratom()
{
  int i,j,i1,i2;
  double onemass;
  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int *mask = atom->mask;
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  double mvv2e = force->mvv2e;
  double massone;
  double vcom [2];  // Temporary array for COM velocty



  if (force->newton) ntotal += atom->nghost;


  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(stress);
    nmax = atom->nmax;
    memory->create(stress,nmax,4,"stress/com/atom:stress");
    array_atom = stress;
  }

  // clear local stress array
  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 4; j++)
      stress[i][j] = 0.0;

  // include kinetic energy term for each atom in group
  // mvv2e converts mv^2 to energy

  for (int n = 0; n < nbondlist; n++)
    {
      // Use the dumbbell COM velocity for both atoms
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
      if (biasflag == NOBIAS)
        {
          vcom[0] = 0.5 * (v[i1][0]+v[i2][0]);
          vcom[1] = 0.5 * (v[i1][1]+v[i2][1]);
        }
      else
        {

          if (mask[i1] & groupbit) temperature->remove_bias(i1,v[i1]);
          if (mask[i2] & groupbit) temperature->remove_bias(i2,v[i2]);
          vcom[0] = 0.5 * (v[i1][0]+v[i2][0]);
          vcom[1] = 0.5 * (v[i1][1]+v[i2][1]);
          if (mask[i1] & groupbit) temperature->restore_bias(i1,v[i1]);
          if (mask[i2] & groupbit) temperature->restore_bias(i2,v[i2]);
        }
      if (mask[i1] & groupbit)
        {
          if (rmass) massone = mvv2e * rmass[i1];
          else massone = mvv2e * mass[type[i1]];
          stress[i1][0] += massone * vcom[0] * vcom[0];
          stress[i1][1] += massone * vcom[0] * vcom[1];
          stress[i1][2] += massone * vcom[1] * vcom[0];
          stress[i1][3] += massone * vcom[1] * vcom[1];
        }
      if (mask[i2] & groupbit)  // Add the same thing to the other atom
        {
          if (rmass) massone = mvv2e * rmass[i2];
          else massone = mvv2e * mass[type[i2]];
          stress[i2][0] += massone * vcom[0] * vcom[0];
          stress[i2][1] += massone * vcom[0] * vcom[1];
          stress[i2][2] += massone * vcom[1] * vcom[0];
          stress[i2][3] += massone * vcom[1] * vcom[1];
        }
    }

  // convert to stress*volume units = -pressure*volume
  double nktv2p = -force->nktv2p;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      stress[i][0] *= nktv2p;
      stress[i][1] *= nktv2p;
      stress[i][2] *= nktv2p;
      stress[i][3] *= nktv2p;
    }
}

/* ---------------------------------------------------------------------- */

int ComputeStressCOMAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = stress[i][0];
    buf[m++] = stress[i][1];
    buf[m++] = stress[i][2];
    buf[m++] = stress[i][3];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeStressCOMAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    stress[j][0] += buf[m++];
    stress[j][1] += buf[m++];
    stress[j][2] += buf[m++];
    stress[j][3] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeStressCOMAtom::memory_usage()
{
  double bytes = nmax*4 * sizeof(double);
  return bytes;
}
