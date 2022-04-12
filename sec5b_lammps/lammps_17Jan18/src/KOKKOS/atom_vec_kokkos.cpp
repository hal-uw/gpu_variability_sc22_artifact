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

#include "atom_vec_kokkos.h"
#include "atom_kokkos.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "atom_masks.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecKokkos::AtomVecKokkos(LAMMPS *lmp) : AtomVec(lmp)
{
  kokkosable = 1;
  buffer = NULL;
  buffer_size = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecKokkos_PackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecKokkos_PackComm(
      const typename DAT::tdual_x_array &x,
      const typename DAT::tdual_xfloat_2d &buf,
      const typename DAT::tdual_int_2d &list,
      const int & iswap,
      const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
      const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
      _x(x.view<DeviceType>()),_list(list.view<DeviceType>()),_iswap(iswap),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {
        const size_t maxsend = (buf.view<DeviceType>().dimension_0()*buf.view<DeviceType>().dimension_1())/3;
        const size_t elements = 3;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
        _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
        _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
        const int j = _list(_iswap,i);
      if (PBC_FLAG == 0) {
          _buf(i,0) = _x(j,0);
          _buf(i,1) = _x(j,1);
          _buf(i,2) = _x(j,2);
      } else {
        if (TRICLINIC == 0) {
          _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
          _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
          _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
        } else {
          _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
          _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
          _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
        }
      }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_kokkos(const int &n,
                                          const DAT::tdual_int_2d &list,
                                          const int & iswap,
                                          const DAT::tdual_xfloat_2d &buf,
                                          const int &pbc_flag,
                                          const int* const pbc)
{
  // Check whether to always run forward communication on the host
  // Choose correct forward PackComm kernel

  if(commKK->forward_comm_on_host) {
    sync(Host,X_MASK);
    if(pbc_flag) {
      if(domain->triclinic) {
        struct AtomVecKokkos_PackComm<LMPHostType,1,1> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackComm<LMPHostType,1,0> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if(domain->triclinic) {
        struct AtomVecKokkos_PackComm<LMPHostType,0,1> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackComm<LMPHostType,0,0> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    sync(Device,X_MASK);
    if(pbc_flag) {
      if(domain->triclinic) {
        struct AtomVecKokkos_PackComm<LMPDeviceType,1,1> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackComm<LMPDeviceType,1,0> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if(domain->triclinic) {
        struct AtomVecKokkos_PackComm<LMPDeviceType,0,1> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackComm<LMPDeviceType,0,0> f(atomKK->k_x,buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  }

	return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecKokkos_PackCommSelf {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_x_array _xw;
  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecKokkos_PackCommSelf(
      const typename DAT::tdual_x_array &x,
      const int &nfirst,
      const typename DAT::tdual_int_2d &list,
      const int & iswap,
      const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
      const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
      _x(x.view<DeviceType>()),_xw(x.view<DeviceType>()),_nfirst(nfirst),_list(list.view<DeviceType>()),_iswap(iswap),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {
        _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
        _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
        const int j = _list(_iswap,i);
      if (PBC_FLAG == 0) {
          _xw(i+_nfirst,0) = _x(j,0);
          _xw(i+_nfirst,1) = _x(j,1);
          _xw(i+_nfirst,2) = _x(j,2);
      } else {
        if (TRICLINIC == 0) {
          _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd;
          _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd;
          _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
        } else {
          _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
          _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
          _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
        }
      }

  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_self(const int &n, const DAT::tdual_int_2d &list, const int & iswap,
                                        const int nfirst, const int &pbc_flag, const int* const pbc) {
  if(commKK->forward_comm_on_host) {
    sync(Host,X_MASK);
    modified(Host,X_MASK);
    if(pbc_flag) {
      if(domain->triclinic) {
      struct AtomVecKokkos_PackCommSelf<LMPHostType,1,1> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      } else {
      struct AtomVecKokkos_PackCommSelf<LMPHostType,1,0> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      }
    } else {
      if(domain->triclinic) {
      struct AtomVecKokkos_PackCommSelf<LMPHostType,0,1> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      } else {
      struct AtomVecKokkos_PackCommSelf<LMPHostType,0,0> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      }
    }
  } else {
    sync(Device,X_MASK);
    modified(Device,X_MASK);
    if(pbc_flag) {
      if(domain->triclinic) {
      struct AtomVecKokkos_PackCommSelf<LMPDeviceType,1,1> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      } else {
      struct AtomVecKokkos_PackCommSelf<LMPDeviceType,1,0> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      }
    } else {
      if(domain->triclinic) {
      struct AtomVecKokkos_PackCommSelf<LMPDeviceType,0,1> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      } else {
      struct AtomVecKokkos_PackCommSelf<LMPDeviceType,0,0> f(atomKK->k_x,nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
      }
    }
  }
	return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnpackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_const _buf;
  int _first;

  AtomVecKokkos_UnpackComm(
      const typename DAT::tdual_x_array &x,
      const typename DAT::tdual_xfloat_2d &buf,
      const int& first):_x(x.view<DeviceType>()),_buf(buf.view<DeviceType>()),
                        _first(first) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      _x(i+_first,0) = _buf(i,0);
      _x(i+_first,1) = _buf(i,1);
      _x(i+_first,2) = _buf(i,2);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_comm_kokkos(const int &n, const int &first,
    const DAT::tdual_xfloat_2d &buf ) {
  if(commKK->forward_comm_on_host) {
    sync(Host,X_MASK);
    modified(Host,X_MASK);
    struct AtomVecKokkos_UnpackComm<LMPHostType> f(atomKK->k_x,buf,first);
    Kokkos::parallel_for(n,f);
  } else {
    sync(Device,X_MASK);
    modified(Device,X_MASK);
    struct AtomVecKokkos_UnpackComm<LMPDeviceType> f(atomKK->k_x,buf,first);
    Kokkos::parallel_for(n,f);
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j,0);
      buf[m++] = h_x(j,1);
      buf[m++] = h_x(j,2);
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j,0) + dx;
      buf[m++] = h_x(j,1) + dy;
      buf[m++] = h_x(j,2) + dz;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_vel(int n, int *list, double *buf,
                                 int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j,0);
      buf[m++] = h_x(j,1);
      buf[m++] = h_x(j,2);
      buf[m++] = h_v(j,0);
      buf[m++] = h_v(j,1);
      buf[m++] = h_v(j,2);
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = h_x(j,0) + dx;
        buf[m++] = h_x(j,1) + dy;
        buf[m++] = h_x(j,2) + dz;
        buf[m++] = h_v(j,0);
        buf[m++] = h_v(j,1);
        buf[m++] = h_v(j,2);
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = h_x(j,0) + dx;
        buf[m++] = h_x(j,1) + dy;
        buf[m++] = h_x(j,2) + dz;
        if (atom->mask[i] & deform_groupbit) {
          buf[m++] = h_v(j,0) + dvx;
          buf[m++] = h_v(j,1) + dvy;
          buf[m++] = h_v(j,2) + dvz;
        } else {
          buf[m++] = h_v(j,0);
          buf[m++] = h_v(j,1);
          buf[m++] = h_v(j,2);
        }
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    h_x(i,0) = buf[m++];
    h_x(i,1) = buf[m++];
    h_x(i,2) = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_comm_vel(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    h_x(i,0) = buf[m++];
    h_x(i,1) = buf[m++];
    h_x(i,2) = buf[m++];
    h_v(i,0) = buf[m++];
    h_v(i,1) = buf[m++];
    h_v(i,2) = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_PackReverse {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array_randomread _f;
  typename ArrayTypes<DeviceType>::t_ffloat_2d _buf;
  int _first;

  AtomVecKokkos_PackReverse(
      const typename DAT::tdual_f_array &f,
      const typename DAT::tdual_ffloat_2d &buf,
      const int& first):_f(f.view<DeviceType>()),_buf(buf.view<DeviceType>()),
                        _first(first) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _buf(i,0) = _f(i+_first,0);
    _buf(i,1) = _f(i+_first,1);
    _buf(i,2) = _f(i+_first,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_reverse_kokkos(const int &n, const int &first,
    const DAT::tdual_ffloat_2d &buf ) {
  if(commKK->reverse_comm_on_host) {
    sync(Host,F_MASK);
    struct AtomVecKokkos_PackReverse<LMPHostType> f(atomKK->k_f,buf,first);
    Kokkos::parallel_for(n,f);
  } else {
    sync(Device,F_MASK);
    struct AtomVecKokkos_PackReverse<LMPDeviceType> f(atomKK->k_f,buf,first);
    Kokkos::parallel_for(n,f);
  }

  return n*size_reverse;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnPackReverseSelf {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array_randomread _f;
  typename ArrayTypes<DeviceType>::t_f_array _fw;
  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;

  AtomVecKokkos_UnPackReverseSelf(
      const typename DAT::tdual_f_array &f,
      const int &nfirst,
      const typename DAT::tdual_int_2d &list,
      const int & iswap):
      _f(f.view<DeviceType>()),_fw(f.view<DeviceType>()),_nfirst(nfirst),_list(list.view<DeviceType>()),_iswap(iswap) {
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    _fw(j,0) += _f(i+_nfirst,0);
    _fw(j,1) += _f(i+_nfirst,1);
    _fw(j,2) += _f(i+_nfirst,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::unpack_reverse_self(const int &n, const DAT::tdual_int_2d &list, const int & iswap,
                                        const int nfirst) {
  if(commKK->reverse_comm_on_host) {
    sync(Host,F_MASK);
    struct AtomVecKokkos_UnPackReverseSelf<LMPHostType> f(atomKK->k_f,nfirst,list,iswap);
    Kokkos::parallel_for(n,f);
    modified(Host,F_MASK);
  } else {
    sync(Device,F_MASK);
    struct AtomVecKokkos_UnPackReverseSelf<LMPDeviceType> f(atomKK->k_f,nfirst,list,iswap);
    Kokkos::parallel_for(n,f);
    modified(Device,F_MASK);
  }
  return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnPackReverse {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array _f;
  typename ArrayTypes<DeviceType>::t_ffloat_2d_const _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;

  AtomVecKokkos_UnPackReverse(
      const typename DAT::tdual_f_array &f,
      const typename DAT::tdual_ffloat_2d &buf,
      const typename DAT::tdual_int_2d &list,
      const int & iswap):
      _f(f.view<DeviceType>()),_list(list.view<DeviceType>()),_iswap(iswap) {
        const size_t maxsend = (buf.view<DeviceType>().dimension_0()*buf.view<DeviceType>().dimension_1())/3;
        const size_t elements = 3;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    _f(j,0) += _buf(i,0);
    _f(j,1) += _buf(i,1);
    _f(j,2) += _buf(i,2);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_reverse_kokkos(const int &n,
                                          const DAT::tdual_int_2d &list,
                                          const int & iswap,
                                          const DAT::tdual_ffloat_2d &buf)
{
  // Check whether to always run reverse communication on the host
  // Choose correct reverse UnPackReverse kernel

  if(commKK->reverse_comm_on_host) {
    struct AtomVecKokkos_UnPackReverse<LMPHostType> f(atomKK->k_f,buf,list,iswap);
    Kokkos::parallel_for(n,f);
    modified(Host,F_MASK);
  } else {
    struct AtomVecKokkos_UnPackReverse<LMPDeviceType> f(atomKK->k_f,buf,list,iswap);
    Kokkos::parallel_for(n,f);
    modified(Device,F_MASK);
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_reverse(int n, int first, double *buf)
{
  if(n > 0)
    sync(Host,F_MASK);

  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = h_f(i,0);
    buf[m++] = h_f(i,1);
    buf[m++] = h_f(i,2);
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_reverse(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    h_f(j,0) += buf[m++];
    h_f(j,1) += buf[m++];
    h_f(j,2) += buf[m++];
  }

  if(n > 0)
    modified(Host,F_MASK);
}
