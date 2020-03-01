function [Z1, Z2, Z3, Z3_Test] = call_project3Domains(filepath, in_filename, out_filename, save_data)
%CALL_PROJECT3DOMAINS Project data of 3 domains into latent space using KEMA RBF kernel
%   Detailed explanation goes here
	addpath('kema_code/')

	data_path = fullfile(filepath, in_filename);
	load(data_path)

	[Z1, Z2, Z3, Z3_Test] = project3Domains(X1, X2, X3, X3_Test, Y1, Y2, Y3, filepath, out_filename, save_data);

end