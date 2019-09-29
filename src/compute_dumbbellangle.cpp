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
#include "compute_dumbbellangle.h"
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
#include <iostream>
#include <iomanip>

using namespace std;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeDumbbellAngle::ComputeDumbbellAngle(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  angles(NULL)

{
  if (narg != 3) error->all(FLERR,"Illegal compute dumbbellangle/atom command");
  peratom_flag = 1;
  size_peratom_cols = 0;
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeDumbbellAngle::~ComputeDumbbellAngle()
{
  memory->destroy(angles);
}

/* ---------------------------------------------------------------------- */

void ComputeDumbbellAngle::compute_peratom()
{

  invoked_peratom = update->ntimestep;

  // grow angles array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(angles);
    nmax = atom->nmax;
    memory->create(angles,nmax,"dumbbellangle/atom:angles");
    vector_atom = angles;
  }


  double **x = atom->x;
  int i1, i2;
  double dx, dy, angle;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  // // clear local stress array
  // for (i = 0; i < ntotal; i++)
  //   for (j = 0; j < 4; j++)
  //     stress[i][j] = 0.0;
  // add in per-atom contributions from each force

  for (int n = 0; n < nbondlist; n++)
    {
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
      dx = x[i2][0] - x[i1][0];
      dy = x[i2][1] - x[i1][1];
      // angle = atan2(abs(dy), abs(dx));
      // angle = abs(dy);
      angle = atan(abs(dy/dx));
      // printf("\nAngle: %2.8f", angle);
      angles[i1] = angle;
      angles[i2] = angle;
    }
}

// /* ---------------------------------------------------------------------- */

// int ComputeDumbbellAngle::pack_reverse_comm(int n, int first, double *buf)
// {
//   int i,m,last;

//   m = 0;
//   last = first + n;
//   for (i = first; i < last; i++) {
//     buf[m++] = stress[i][0];
//     buf[m++] = stress[i][1];
//     buf[m++] = stress[i][2];
//     buf[m++] = stress[i][3];
//   }
//   return m;
// }

// /* ---------------------------------------------------------------------- */

// void ComputeDumbbellAngle::unpack_reverse_comm(int n, int *list, double *buf)
// {
//   int i,j,m;

//   m = 0;
//   for (i = 0; i < n; i++) {
//     j = list[i];
//     stress[j][0] += buf[m++];
//     stress[j][1] += buf[m++];
//     stress[j][2] += buf[m++];
//     stress[j][3] += buf[m++];
//   }
// }

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeDumbbellAngle::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
