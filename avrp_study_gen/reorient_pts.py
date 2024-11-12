import numpy as np
import vtk
import vtk.util.numpy_support as v2n
from scipy.optimize import minimize
import mesh_helpers as mh
import argparse

def trans(translation):
  T = np.eye(4)
  T[:3, 3] = translation
  return T

def rotz(angle):
  c, s = np.cos(angle), np.sin(angle)
  R = np.array([[c, -s, 0, 0],
          [s,  c, 0, 0],
          [0,  0, 1, 0],
          [0,  0, 0, 1]])
  return R

def tform_apply(points, T):
  points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
  transformed_points = points_hom.dot(T.T)[:, :3]
  return transformed_points

def rotxy_apply(vector):
  # Placeholder for actual implementation
  Tx = np.eye(4)
  Ty = np.eye(4)
  return Tx, Ty, None, None

def create_mesh_from_points(points):
  points_vtk = vtk.vtkPoints()
  for point in points:
    points_vtk.InsertNextPoint(point)
  
  polydata = vtk.vtkPolyData()
  polydata.SetPoints(points_vtk)
  return polydata

def reorient_pts_generic(fnMesh, labelSTJ, labelLVO, labelIAS):
  m = mh.read_polydata(fnMesh)

  labels = v2n.vtk_to_numpy(m.GetPointData().GetScalars())

  print(labels)

  stj_ind = np.where(labels == labelSTJ)[0]
  lvo_ind = np.where(labels == labelLVO)[0]
  septum_ind = np.where(labels == labelIAS)[0]

  print(stj_ind)

  points = v2n.vtk_to_numpy(m.GetPoints().GetData())

  # get the outflow vector
  mean_stj = np.mean(points[stj_ind], axis=0)
  mean_lvo = np.mean(points[lvo_ind], axis=0)
  outflow_vec = np.array([mean_lvo, mean_stj])

  print("outflow_vec: ", outflow_vec)

  # compose a translation tranform to move in the reverse of outflow_vec
  tform_translation = trans(-outflow_vec[0])
  print("tform_translation: ", tform_translation)


  pts_origin = tform_apply(vtk.util.numpy_support.vtk_to_numpy(m.GetPoints().GetData()), Tt)
  outflow_vec_trans = tform_apply(outflow_vec, Tt)

  Tx, Ty, _, _ = rotxy_apply(outflow_vec_trans[1])

  pts_rotxy = tform_apply(tform_apply(pts_origin, Tx), Ty)

  m_new = np.mean(pts_rotxy[septum_ind], axis=0)
  f_new = np.array([100, 0, 0])

  def rotz_apply(x, f_new, m_new):
    Tz = rotz(x)
    transformed = tform_apply(np.array([m_new]), Tz)[0]
    return np.linalg.norm(transformed - f_new)

  res = minimize(rotz_apply, 0, args=(f_new, m_new))
  gamma = res.x[0]
  Tz = rotz(gamma)

  pts_rotxyz = tform_apply(pts_rotxy, Tz)

  T = Tz.dot(Ty).dot(Tx).dot(Tt)

  return create_mesh_from_points(pts_rotxyz)


def main():
  parser = argparse.ArgumentParser(description='Reorient points in a mesh.')
  parser.add_argument('-i', '--input', type=str, required=True, help='Input mesh filename')
  parser.add_argument('-o', '--output', type=str, required=True, help='Output mesh filename')
  args = parser.parse_args()

  mesh = reorient_pts_generic(args.input, 6, 5, 7)

  mh.write_polydata(mesh, args.output)


if __name__ == '__main__':
  main()