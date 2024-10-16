function p = vtk_polydata_read(file, varargin)
    % Read VTK polydata into a struct
    % Usage:
    %   p = vtk_polydata_read(file, pars)
    % Parameters:
    %   pars.encoding           One of 'ieee-le', 'ieee-be' (default)
    % Return Value:
    %   p.hdr                   Header information
    %   p.points                N x 3 array of point coordinates
    %   p.cells.(name)          cell array of polygons, lines, etc
    %   p.point_data            Point data arrays, following format:
    %   p.point_data(i).name    Array name
    %   p.point_data(i).type    VTK type (normals,  vectors, field)
    %   p.point_data(i).data    N x K array of values
    
        % Initialize the parameters
        if(nargin > 1) pars = varargin{1}; else pars = struct(); end
        if(~isfield(pars,'encoding')) pars.encoding='ieee-be'; end        
    
        % Open file
        fid = fopen(file, 'r');
    
        % Read the first line
        % # vtk DataFile Version x.x
        firstLine = textscan(fid, '# %s %s %s %[^\n]\n', 1);
        fileVersion = char(firstLine{4});
        crInd = regexp(fileVersion, '\r', 'once'); %trying to detect CR
    
        lb = '\n'; %default line break is LF only
        
        if (~isempty(crInd))
            lb = '\r\n'; %line break is CRLF
        end
    
        strPat = "%s" + lb;
    
        fileVersion = str2double(fileVersion);
    
        if (isnan(fileVersion))
            warning("Version cannot be read! Setting version to default value 4.2.");
            fileVersion = 4.2;
        end
    
        isVersion51Plus = fileVersion >= 5.1;
    
        % Read the header (2 lines)
        p.hdr.name = vtkreadstr(fid, "%[^" + lb + "]" + lb);
        p.hdr.type = vtkreadstr(fid, strPat);
          
        % Read the dataset type
        p.hdr.dst = vtkreadstr(fid, 'DATASET ' + strPat);
        if ~strcmpi(p.hdr.dst, 'POLYDATA')
            error('Data types other than POLYDATA unsupported');
        end
        
        % Read the rest of the stuff in the file
        mode = 'intro';
        nattr = 0;
        
        while ~feof(fid)
            
            % Get the keyword (first word of the line)
            key = vtkreadstr(fid, '%s');        
            
            if (length(key) == 0) continue; end
            
            % fprintf('Reading %s\n', vtk_decode(key));                    
                
            if strcmp(mode,'intro') && strcmpi(key, 'points')
                
                % Read the number of points and the type
                dat = vtkread(fid, "%d %s" + lb, 1);
                n = double(dat{1});
                data_type = char(dat{2});
    
                % Read point values
                X = vtkreaddata(fid, p, 3 * n, data_type, pars);
                p.points = reshape(X, 3, [])';
    
            elseif strcmp(mode,'intro') && any(strcmpi(key, ...
                    {'polygons','vertices','lines','triangle_strips'}))
    
                % Read size information
                dat = vtkread(fid, "%d %d" + lb, 1);
    
                if (isVersion51Plus)
    
                    nOffset = double(dat{1}); % Read offset array size
                    nConn = double(dat{2}); % Read connectivity array size
                    mode = lower(key); % Change mode to use offset-conn logic
    
                else
    
                    % Read the number of cells and storage size
                    n = double(dat{1});
                    storage = double(dat{2});
                    data_type = 'uint';            
                    
                    % Read the cell data
                    T = vtkreaddata(fid, p, storage, data_type, pars);
                    
                    % Connect the cell data together
                    i = 1; j = 1;
                    while i <= storage
                        cd{j} = T(i+1: i+T(i)) + 1;
                        j = j + 1;
                        i = i + T(i) + 1;
    
                    end
                   
                % Place in the appropriate array
                p.cells.(lower(key)) = cd;
    
                end
            
            elseif any(strcmpi(mode, ... 
                    {'polygons','vertices','lines','triangle_strips'})) && ...
                    any(strcmpi(key, {'offsets', 'connectivity'}))
                
                % Offset-Connectivity logic for version >= 5.1
    
                data_type = vtkreadstr(fid, strPat);
    
                if (strcmpi(key, 'offsets'))
                    offsets = vtkreaddata(fid, p, nOffset, data_type, pars);
                    continue;
                end
    
                if (strcmpi(key, 'connectivity'))
    
                    if (isempty(offsets) && nConn ~= 0)
                        error("Cannot process connectivity of " + mode ...
                            + ". Missing offsets information!");
                    end
    
                    connArray = vtkreaddata(fid, p, nConn, data_type, pars);
    
                    offsetInd = 1; % index iterating the offset array
                    
                    while (offsetInd < nOffset)
                        crnt = offsets(offsetInd); % position reading starts
                        next = offsets(offsetInd + 1); % position reading should end before
    
                        cd{offsetInd} = connArray(crnt + 1 : next) + 1; % +1 because points are using 1-based index
    
                        offsetInd = offsetInd + 1;
                    end
    
                    % Place in the appropriate array
                    p.cells.(lower(mode)) = cd;
                    
                    offsets = []; % reset offset array
                    mode = 'intro'; % reset mode to intro
                    continue;
    
                end
                
            elseif any(strcmpi(key, {'point_data', 'cell_data'}))
                
                % Set the mode variable
                mode = lower(key);
                
                % Initialize the data arrays
                % p.(mode) = [];
                
                % Reset the field counter
                n_fields = 0;
                
                % Read the number of attributes
                nattr = vtkreadnum(fid, "%d" + lb);
                            
            elseif any(strcmpi(mode, {'point_data', 'cell_data'}))
                if (n_fields > 0)
                    % Read the field information
                    arr.name = vtk_decode(key);
                    arr.type = 'field';
                    ncomp = vtkreadnum(fid, '%d');
                    ntuples = vtkreadnum(fid, '%d');
                    data_type = vtkreadstr(fid, strPat);
                    
                    % Read the tuple data
                    X = vtkreaddata(fid, p, ncomp * ntuples, data_type, pars);
                    arr.data = reshape(X, ncomp, [])';
                                    
                    % Append the array
                    if ~isfield(p, mode)
                        p.(mode)(1) = arr;
                    else
                        p.(mode)(1+length(p.(mode))) = arr;
                    end
                    
                    % Decrement the field counter
                    n_fields = n_fields - 1;
                    continue;
                end
                
                if any(strcmpi(key, {'normals','scalars','color_scalars',...
                    'vectors', 'texture_coordinates', 'tensors'}))
                
                    % Create an array
                    arr.name = vtk_decode(vtkreadstr(fid, '%s'));
                    arr.type = lower(key);
    
                    % We are reading some sort of attributes
                    if any(strcmpi(key, {'normals','vectors'}))
    
                        % Read the data
                        data_type = vtkreadstr(fid, strPat);
                        X = vtkreaddata(fid, p, nattr * 3, data_type, pars);
                        arr.data = reshape(X, 3, [])';
    
                    elseif strcmpi(key, 'tensors')
    
                        % Read the data
                        data_type = vtkreadstr(fid, strPat);
                        X = vtkreaddata(fid, p, nattr * 9, data_type, pars);
                        arr.data = reshape(X, 9, [])';
    
                    elseif strcmpi(key, 'texture_coordinates')
    
                        % Read the data
                        ncomp = vtkreadnum(fid, '%d');
                        data_type = vtkreadstr(fid, strPat);
                        X = vtkreaddata(fid, p, nattr * ncomp, data_type, pars);
                        arr.data = reshape(X, ncomp, [])';
    
                    elseif strcmpi(key, 'scalars')
                        
                        % Read the scarar data
                        data_type = vtkreadstr(fid, strPat);
                        junk = vtkreadstr(fid, "%s %s" + lb);
                        X = vtkreaddata(fid, p, nattr, data_type, pars);
                        arr.data = reshape(X, 1, [])';
                        
                    else
                        
                        error('Dataformat %s is not yet supported', key);
    
                    end
    
                    % Append the array
                    if ~isfield(p, mode)
                        p.(mode)(1) = arr;
                    else
                        p.(mode)(1+length(p.(mode))) = arr;
                    end
                
                elseif strcmpi(key, 'field')
                    
                    % Enter field reading mode
                    vtkreadstr(fid, '%s');
                    n_fields = vtkreadnum(fid, "%d" + lb);
                    
                else
    
                    error('Unknown entry %s', key);
                    
                end
                
            else
                
                error('Unknown entry %s', key);
    
            end
        end
    
    
        fclose(fid);
    end
    
    function s = vtkread(fid, str, n)
        s = textscan(fid, str, n, 'ReturnOnError', 0, 'CommentStyle', '#');
    end
    
    function s = vtkreadstr(fid, pat)
        str = textscan(fid, pat, 1, 'ReturnOnError', 0, 'CommentStyle', '#');
        s = char(str{1});
    end
    
    function n = vtkreadnum(fid, pat)
        val = textscan(fid, pat, 1, 'ReturnOnError', 0, 'CommentStyle', '#');
        n = double(val{1});
    end
    
    function X = vtkreaddata(fid, p, comp, data_type, pars)
        if strcmpi(p.hdr.type, 'ascii')
            car = textscan(fid, '%f', comp, ...
                'ReturnOnError', 0, 'CommentStyle', '#');
            X = car{1};
        else
            if strcmpi(data_type, 'float')
                conv = 'float32=>double';
            elseif strcmpi(data_type, 'double')
                conv = 'float64=>double';
            elseif strcmpi(data_type, 'uint')
                conv = 'uint32=>uint32';
            elseif strcmpi(data_type, 'vtkIdType')
                conv = 'uint32=>uint32';
            elseif strcmpi(data_type, 'vtktypeint64')
                conv = 'int64=>int64';
            else
                error('data_type %s is unsupported', data_type);
            end
                
            X = fread(fid, comp, conv, pars.encoding);
        end
    end
        
    function S = vtk_decode(t)
        S = strrep(t,'%20',' ');
    end