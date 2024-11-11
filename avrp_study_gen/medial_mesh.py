import numpy as np
import mesh_helpers as mh
import argparse
from scipy.optimize import minimize
from vtk.util.numpy_support import vtk_to_numpy

def reorient_pts(polydata, labelSTJ, labelLVO, labelIAS):
  labels_vtk = polydata.GetPointData().GetArray('Label')
  labels = vtk_to_numpy(labels_vtk)
  points = vtk_to_numpy(polydata.GetPoints().GetData())

  stj_ind = np.where(labels == labelSTJ)[0]
  lvo_ind = np.where(labels == labelLVO)[0]
  ias_ind = np.where(labels == labelIAS)[0]

  outflow_vec = np.array([np.mean(points[lvo_ind, :], axis=0), np.mean(points[stj_ind, :], axis=0)])

  print(outflow_vec)

  Tt = mh.trans(-outflow_vec[0, :])
  pts_origin = mh.tform_apply(points, Tt)

  print(pts_origin)

  outflow_vec_trans = mh.tform_apply(outflow_vec, Tt)

  Tx, Ty, _, _ = mh.rotxy_apply(outflow_vec_trans[1, :])

  pts_rotxy = mh.tform_apply(mh.tform_apply(pts_origin, Tx), Ty)

  m_new = np.mean(pts_rotxy[ias_ind, :], axis=0)
  f_new = np.array([100, 0, 0])

  def objective(x):
    return mh.rotz_apply(x, f_new, m_new)

  res = minimize(objective, 0)
  gamma = res.x[0]
  Tz = mh.rotz(gamma)

  pts_rotxyz = mh.tform_apply(pts_rotxy, Tz)

  T = Tz @ Ty @ Tx @ Tt

  return pts_rotxyz, T

def get_medial_mesh(labelMesh, labelSTJ, labelLVO, labelIAS):
  # Convert labels to numbers
  labelSTJ = int(labelSTJ)
  labelLVO = int(labelLVO)
  labelIAS = int(labelIAS)

  # Apply rigid transform to orient the outflow tract along the vertical axis
  pts_rotxyz, T_init = reorient_pts(labelMesh, labelSTJ, labelLVO, labelIAS)

  # Obtain medial and boundary meshes from the reoriented point cloud
  _, mbnd, mmed = mh.edge_sampling(pts_rotxyz)

  # Apply inverse rigid transform to medial and boundary points so it's aligned with the original segmentation
  T_init_inv = np.linalg.inv(T_init)
  mmed.points = mh.tform_apply(mmed.points, T_init_inv)
  mbnd.points = mh.tform_apply(mbnd.points, T_init_inv)

  return mmed, mbnd

def main():
  # parse arguments:
  # -i <input label mesh> -om <output medial mesh> -ob <output boundary mesh> -stj <labelSTJ> -lvo <labelLVO> -ias <labelIAS>
  # e.g. python medial_mesh.py -i label_mesh.vtk -om medial_mesh.vtk -ob boundary_mesh.vtk -stj 1 -lvo 2 -ias 3

  parser = argparse.ArgumentParser(description='Generate medial and boundary meshes from a labeled mesh.')
  parser.add_argument('-i', '--input', required=True, help='Input label mesh file')
  parser.add_argument('-om', '--output_medial', required=True, help='Output medial mesh file')
  parser.add_argument('-ob', '--output_boundary', required=True, help='Output boundary mesh file')
  parser.add_argument('-stj', '--labelSTJ', required=True, type=int, help='Label for STJ')
  parser.add_argument('-lvo', '--labelLVO', required=True, type=int, help='Label for LVO')
  parser.add_argument('-ias', '--labelIAS', required=True, type=int, help='Label for IAS')

  args = parser.parse_args()

  # read input label mesh
  labelMesh = mh.read_polydata(args.input)

  # get medial and boundary meshes
  meshMedial, meshBoundary = get_medial_mesh(labelMesh, args.labelSTJ, args.labelLVO, args.labelIAS)

  # write medial and boundary meshes
  mh.write_polydata(args.output_medial, meshMedial)
  mh.write_polydata(args.output_boundary, meshBoundary)

if __name__ == "__main__":
  main()