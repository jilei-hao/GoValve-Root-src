function [] = reorient_pts_generic(fnvtk, labelSTJ, labelLVO, labelIAS)

  % read medial model and point labels
  m = vtk_polydata_read(fnvtk);
  labels = m.point_data(1).data;
  
  
  % Display the data types and values for debugging
  
  stj_ind = find(labels == labelSTJ);
  lvo_ind = find(labels == labelLVO);
  septum_ind = find(labels == labelIAS);
  
  % outflow tract vector
  outflow_vec = [mean(m.points(lvo_ind,:)); mean(m.points(stj_ind,:))];

  disp('Outflow vector:');
  disp(size(outflow_vec));
  
  % translate point clouds to origin
  Tt = trans(-outflow_vec(1,:));
  pts_origin = tform_apply(m.points,Tt);
  outflow_vec_trans = tform_apply(outflow_vec,Tt);
  
  % rotate eigenvector associated with lowest variance to align it with the
  % positive z-axis
  
  [Tx,Ty,~,~] = rotxy_apply(outflow_vec_trans(2,:)');
  
  % now rotate points
  pts_rotxy = tform_apply(tform_apply(pts_origin,Tx),Ty);
  
  % next rotate the model around the z-axis
  m_new = mean(pts_rotxy(septum_ind,:));
  f_new = [100 0 0];
  
  x0 = 0; 
  f = @(x)rotz_apply(x,f_new,m_new);
  x = fminunc(f,x0);
  
  gamma = x;
  Tz = rotz(gamma);
  
  % finally rotate the template
  pts_rotxyz = tform_apply(pts_rotxy,Tz);


  
  % composite transform
  T = Tz * Ty * Tx * Tt;

  mesh = create_mesh_from_points(pts_rotxyz);
  vtk_polydata_write("/home/jileihao/data/avrs/studies/bavcta005/scan2/debug/pts_reorient.vtk", mesh);
end


function pts_new = tform_apply(pts,T)

  pts = [pts'; ones(1,size(pts,1))];
  pts_new = T*pts;
  pts_new = pts_new(1:3,:)';
  
  end
  
  % -------------------------------------------------------------------------
  
function Tx = rotx(alpha)

  Tx = [1     0           0           0;
        0     cos(alpha)  -sin(alpha) 0;
        0     sin(alpha)  cos(alpha)  0;
        0     0           0           1];

end
  
  % -------------------------------------------------------------------------
  
function Ty = roty(beta)

  Ty = [cos(beta)    0    sin(beta)   0;
        0            1    0           0;
        -sin(beta)   0    cos(beta)   0;
        0            0    0           1];
  
end
  
  % -------------------------------------------------------------------------
  
function Tz = rotz(gamma)

  Tz = [cos(gamma)    -sin(gamma) 0   0;
        sin(gamma)    cos(gamma)  0   0;
        0             0           1   0;
        0             0           0   1];

end
  
  % -------------------------------------------------------------------------
  
function Tt = trans(x)

Tt = [1 0   0   x(1);
      0 1   0   x(2);
      0 0   1   x(3);
      0 0   0   1 ];
end
  
  % -------------------------------------------------------------------------
  
  function [Tx,Ty,alpha,beta] = rotxy_apply(v)
  
  % Angle of rotation about x-axis
  alpha = acos(v(3)/sqrt(v(2)^2 + v(3)^2));
  if v(2) < 0
      alpha = -alpha;
  end
  
  Tx = rotx(alpha);
   
  v_rx = Tx*[v; 1];
  
  % Angle of rotation about y-axis
  beta = acos(v_rx(3)/sqrt(v_rx(1)^2 + v_rx(3)^2));
  if v_rx(1) > 0
      beta = -beta;
  end
  
  Ty = roty(beta);
    
  v_rxy = Ty * v_rx;
  
  % Check that rotation is correct
  if (abs(v_rxy(1)) < 1E-6) && (abs(v_rxy(2)) < 1E-6) && ((abs(v_rxy(3))-1) < 1E-6)
      disp('Third eigenvector is correctly aligned');
  else
      disp('Check third eigenvector');
  end
  
  end
  
  % -------------------------------------------------------------------------
  function D = rotz_apply(x,f,m)
  
  % Rotation angle and transform
  gamma = x;
  Tz = rotz(gamma);
  
  % Transformed points
  m_new = tform_apply(m,Tz);
  
  D = (f - m_new).*(f - m_new);
  D = sum(D(:));
  
  end

function m = create_mesh_from_points(points)
  m = struct;
  m.hdr.name = 'File written by itkPolyDataMeshIO';
  m.hdr.type = 'ASCII';
  m.hdr.dst = 'POLYDATA';
  m.points = points;
  m.cells = struct;
end
  