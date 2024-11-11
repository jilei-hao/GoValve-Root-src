import numpy as np
import vtk
from scipy.ndimage import binary_closing, gaussian_filter
from skimage.measure import find_contours
from scipy.interpolate import splprep, splev

def tform_apply(pts, T):
  pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  pts_new = np.dot(T, pts.T).T
  return pts_new[:, :3]


def rotx(alpha):
  Tx = np.array([
    [1, 0, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha), 0],
    [0, np.sin(alpha), np.cos(alpha), 0],
    [0, 0, 0, 1]
  ])
  return Tx


def roty(beta):
  Ty = np.array([
    [np.cos(beta), 0, np.sin(beta), 0],
    [0, 1, 0, 0],
    [-np.sin(beta), 0, np.cos(beta), 0],
    [0, 0, 0, 1]
  ])
  return Ty


def rotz(gamma):
  cos_gamma = np.cos(gamma)
  sin_gamma = np.sin(gamma)
  Tz = np.array([
      [cos_gamma, -sin_gamma, 0, 0],
      [sin_gamma, cos_gamma, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
  ])
  return Tz

def rotz_apply(gamma, f_new, m_new):
  Tz = rotz(gamma)
  return np.linalg.norm(np.dot(Tz[:3, :3], f_new) - m_new)



def trans(x):
  Tt = np.array([
    [1, 0, 0, x[0]],
    [0, 1, 0, x[1]],
    [0, 0, 1, x[2]],
    [0, 0, 0, 1]
  ])
  return Tt


def rotxy_apply(v):
  alpha = np.arccos(v[2] / np.sqrt(v[1]**2 + v[2]**2))
  if v[1] < 0:
    alpha = -alpha

  Tx = rotx(alpha)
  v_rx = np.dot(Tx, np.append(v, 1))

  beta = np.arccos(v_rx[2] / np.sqrt(v_rx[0]**2 + v_rx[2]**2))
  if v_rx[0] > 0:
    beta = -beta

  Ty = roty(beta)
  v_rxy = np.dot(Ty, v_rx)

  if (abs(v_rxy[0]) < 1E-6) and (abs(v_rxy[1]) < 1E-6) and (abs(v_rxy[2]) - 1 < 1E-6):
    print('Third eigenvector is correctly aligned')
  else:
    print('Check third eigenvector')

  return Tx, Ty, alpha, beta


def edge_sampling(pts_rotxyz):
  nsamp = 48
  gamma = np.deg2rad(np.arange(0, 360, 10))
  nrot = len(gamma)

  bsh = ((nsamp - 2) // 2 - 1) // 2 + 1
  c1 = 1
  e1 = c1 + bsh
  c2 = e1 + bsh
  e2 = c2 + bsh
  nqs = bsh + 1

  mi_ref_ord = np.hstack((np.arange(1, nsamp + 1).reshape(-1, 1), 
              np.hstack((np.arange(e1, 1, -1), np.arange(1, c2 + 1), np.arange(c2 - 1, e1, -1))).reshape(-1, 1)))

  pts_resamp = np.zeros((nsamp, 3, nrot))

  for i in range(nrot):
    rg = gamma[i]
    Tz = rotz(rg)
    Tz_inv = rotz(-rg)
    pts_rotz = tform_apply(pts_rotxyz, Tz)

    slice_ind = np.where(np.abs(pts_rotz[:, 1]) < 1)[0]
    pts_slab = pts_rotz[slice_ind, :]
    pts_slab_pos = pts_slab[pts_slab[:, 0] > 0, :]

    buff_vertical = 10
    pts_slab_pos[:, 2] += buff_vertical
    pts_slab_pos *= 10

    buff_pad = 10
    xrange = [1, int(np.ceil(np.max(pts_slab_pos[:, 0]) + buff_pad))]
    zrange = [1, int(np.ceil(np.max(pts_slab_pos[:, 2]) + buff_pad))]

    img = np.zeros((zrange[1], xrange[1]))
    z_indices = np.round(pts_slab_pos[:, 2]).astype(int)
    x_indices = np.round(pts_slab_pos[:, 0]).astype(int)

    z_indices = np.clip(z_indices, 0, img.shape[0] - 1)
    x_indices = np.clip(x_indices, 0, img.shape[1] - 1)

    img[z_indices, x_indices] = 1

    img_cl = binary_closing(img, structure=np.ones((50, 50)))
    img_cl_sm = gaussian_filter(img_cl.astype(float), sigma=7)

    contours = find_contours(img_cl_sm, 0.5)
    if len(contours) == 0:
      continue
    cuv = contours[0].T

    if not np.all(np.diff(cuv[0]) >= 0):
      cuv = np.fliplr(cuv)

    emax = np.argmax(cuv[1])
    emin = np.argmin(cuv[1])

    cuv_tcs = np.roll(cuv, -emax, axis=1)
    cuv_tcs = np.hstack((cuv_tcs, cuv_tcs[:, 0].reshape(2, 1)))

    emax_cuv_tcs = 0
    emin_cuv_tcs = emin - emax if emax <= emin else emin + len(cuv[1]) - emax

    mean_pt = np.mean(cuv_tcs, axis=1)
    d = np.abs(cuv_tcs[1] - mean_pt[1])
    c1_cuv_tcs = np.argmin(d[emax_cuv_tcs:emin_cuv_tcs]) + emax_cuv_tcs
    c2_cuv_tcs = np.argmin(d[emin_cuv_tcs:]) + emin_cuv_tcs

    def spline_fit(x, y, nqs):
      tck, _ = splprep([x, y], s=0)
      u_fine = np.linspace(0, 1, nqs)
      return np.array(splev(u_fine, tck)).T

    fap1 = spline_fit(cuv_tcs[0, c1_cuv_tcs:emin_cuv_tcs + 1], cuv_tcs[1, c1_cuv_tcs:emin_cuv_tcs + 1], nqs)
    fap2 = spline_fit(cuv_tcs[0, emin_cuv_tcs:c2_cuv_tcs + 1], cuv_tcs[1, emin_cuv_tcs:c2_cuv_tcs + 1], nqs)
    fap3 = spline_fit(cuv_tcs[0, c2_cuv_tcs:], cuv_tcs[1, c2_cuv_tcs:], nqs)
    fap4 = spline_fit(cuv_tcs[0, emax_cuv_tcs:c1_cuv_tcs + 1], cuv_tcs[1, emax_cuv_tcs:c1_cuv_tcs + 1], nqs)

    fap = np.vstack((fap1, fap2[1:], fap3[1:], fap4[1:-1]))

    pts_slice = np.hstack((fap[:, 0].reshape(-1, 1), np.zeros((len(fap), 1)), fap[:, 1].reshape(-1, 1)))
    pts_zrot_inv = tform_apply(pts_slice, Tz_inv)
    pts_resamp[:, :, i] = pts_zrot_inv / 10
    pts_resamp[:, 2, i] -= buff_vertical

  pts_resamp = np.concatenate((pts_resamp, np.zeros((nsamp, 1, nrot))), axis=1)
  for i in range(nrot):
    pts_resamp[:, 3, i] = np.arange(i * nsamp, (i + 1) * nsamp)

  pts_resamp_list = pts_resamp.reshape(-1, 4)
  medind = np.zeros(nsamp * nrot)
  triangles_bnd = []
  bnd_side = []

  for s in range(nrot):
    i = nrot - 1 if s == 0 else s - 1
    j = s

    rstart = s * nsamp
    medind[rstart:rstart + nsamp] = mi_ref_ord[:, 1] + s * ((nsamp - 2) // 2 + 2)

    for n in range(nsamp):
      if n == e1:
        if e1 % 2 == 1:
          t = [[pts_resamp[n, 3, i], pts_resamp[n - 1, 3, j], pts_resamp[n, 3, j]],
             [pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, j]]]
        else:
          t = [[pts_resamp[n, 3, i], pts_resamp[n - 1, 3, i], pts_resamp[n, 3, j]],
             [pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, i]]]
        bside = [1, 0]
      elif n == c1:
        t = [[pts_resamp[n, 3, i], pts_resamp[nsamp - 1, 3, j], pts_resamp[n, 3, j]],
           [pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, j]]]
        bside = [1, 1]
      elif n == e2:
        if e2 % 2 == 1:
          t = [[pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, j]],
             [pts_resamp[n, 3, i], pts_resamp[n - 1, 3, j], pts_resamp[n, 3, j]]]
        else:
          t = [[pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, i]],
             [pts_resamp[n, 3, i], pts_resamp[n - 1, 3, i], pts_resamp[n, 3, j]]]
        bside = [1, 0]
      elif n == nsamp - 1:
        t = [[pts_resamp[n, 3, i], pts_resamp[n - 1, 3, i], pts_resamp[n, 3, j]],
           [pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[0, 3, i]]]
        bside = [1, 1]
      else:
        if n % 2 == 1:
          t = [[pts_resamp[n, 3, i], pts_resamp[n - 1, 3, j], pts_resamp[n, 3, j]],
             [pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, j]]]
        else:
          t = [[pts_resamp[n, 3, i], pts_resamp[n - 1, 3, i], pts_resamp[n, 3, j]],
             [pts_resamp[n, 3, i], pts_resamp[n, 3, j], pts_resamp[n + 1, 3, i]]]

        if n < c2:
          bside = [1, 1] if n < e1 else [0, 0]
        else:
          bside = [0, 0] if n < e2 else [1, 1]

      triangles_bnd.extend(t)
      bnd_side.extend(bside)

  pts_resamp_list = pts_resamp_list[:, :3]
  triangles_bnd = np.array(triangles_bnd)
  bnd_side = np.array(bnd_side)

  return pts_resamp_list, triangles_bnd, bnd_side, medind

def read_polydata(fn):
  if fn.endswith('.vtk'):
    reader = vtk.vtkPolyDataReader()
  elif fn.endswith('.vtp'):
    reader = vtk.vtkXMLPolyDataReader()
  else:
    raise ValueError('Unknown file format')
  
  reader.SetFileName(fn)
  reader.Update()
  polydata = reader.GetOutput()
  return polydata

def write_polydata(fn, data):
  if fn.endswith('.vtk'):
    writer = vtk.vtkPolyDataWriter()
  elif fn.endswith('.vtp'):
    writer = vtk.vtkXMLPolyDataWriter()
  else:
    raise ValueError('Unknown file format')
  
  writer.SetFileName(fn)
  writer.SetInputData(data)
  writer.Write()