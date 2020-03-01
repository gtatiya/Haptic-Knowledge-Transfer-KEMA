function [Z1, Z2, Z3, Z3_Test] = project3Domains(X1, X2, X3, X3_Test, Y1, Y2, Y3, filepath, out_filename, save_data)
%PROJECT3DOMAINS Project data of 3 domains into latent space using KEMA RBF kernel
%   Remember that labels start from 1, Label 0 means no label.

	options.graph.nn = 10;  %KNN graph number of neighbors
	mu = 0.5;               %(1-mu)*L  + mu*(Ls)
	NF = 100; %GT: Max. no. of latent features allowed

	%% KEMA - RBF KERNEL
	disp('Mapping with the RBF kernel ...')

	X1 = X1';
	X2 = X2';
	X3 = X3';
    X3_Test = X3_Test';
	Y = [Y1;Y2;Y3;];
	[d1, n1] = size(X1);
	[d2, n2] = size(X2);
	[d3, n3] = size(X3);

	% 2) Compute RBF kernels
	% pdist: Pairwise distance between pairs of observations
	kernel_name = 'rbf'; %lin, rbf, sam, chi2
	% 1st domain
	sigma1 = 15*mean(pdist(X1'));
	K1 = kernelmatrix(kernel_name,[X1],[X1],sigma1);

	% 2nd domain
	sigma2 = 15*mean(pdist(X2'));
	K2 = kernelmatrix(kernel_name,[X2],[X2],sigma2);

	% 3rd domain
	sigma3 = 15*mean(pdist(X3'));
	K3 = kernelmatrix(kernel_name,[X3],[X3],sigma3);

	% blkdiag: Block diagonal matrix
	K = blkdiag(K1,K2,K3);
    
    KT3 = kernelmatrix(kernel_name,[X3],[X3_Test],sigma3);
    
	%%%%%%% COMPUTE A AND B

	% 2) graph Laplacians
	G1 = buildKNNGraph([X1]',options.graph.nn,1);
	G2 = buildKNNGraph([X2]',options.graph.nn,1);
	G3 = buildKNNGraph([X3]',options.graph.nn,1);
	W = blkdiag(G1,G2,G3);
	W = double(full(W));
	clear G*

	% Class Graph Laplacian
	Ws = repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))'; Ws(Y == 0,:) = 0; Ws(:,Y == 0) = 0; Ws = double(Ws);
	Wd = repmat(Y,1,length(Y)) ~= repmat(Y,1,length(Y))'; Wd(Y == 0,:) = 0; Wd(:,Y == 0) = 0; Wd = double(Wd);

	Sws = sum(sum(Ws));
	Sw = sum(sum(W));
	Ws = Ws/Sws*Sw;

	Swd = sum(sum(Wd));
	Wd = Wd/Swd*Sw;

	Ds = sum(Ws,2); Ls = diag(Ds) - Ws;
	Dd = sum(Wd,2); Ld = diag(Dd) - Wd;
	D = sum(W,2); L = diag(D) - W;

	% Tune the generalized eigenproblem
	A = ((1-mu)*L  + mu*(Ls)); % (n1+n2) x (n1+n2) %  
	B = Ld;         % (n1+n2) x (n1+n2) %        
	%%%%%%%

	KAK = K*A*K;
	KBK = K*B*K;

	% 3) Extract all features (now we can extract n dimensions!)
	[ALPHA, LAMBDA] = gen_eig(KAK,KBK,'LM');
	[LAMBDA, j] = sort(diag(LAMBDA));
	ALPHA = ALPHA(:,j);

	% 4) Project the data
	nVectRBF = min(NF,rank(KBK));
	nVectRBF = min(nVectRBF,rank(KAK));

	E1 = ALPHA(1:n1, 1:nVectRBF);
	E2 = ALPHA(n1+1:n1+n2, 1:nVectRBF);
	E3 = ALPHA(n1+n2+1:end, 1:nVectRBF);

	Phi1toF = E1'*K1;
	Phi2toF = E2'*K2;
	Phi3toF = E3'*K3;
    
    Phi3TtoF = E3'*KT3;

	% 5) IMPORTAT: Normalize!!!!
	m3 = mean(Phi3toF');
	s3 = std(Phi3toF');

	Phi1toF = zscore(Phi1toF')';
	Phi2toF = zscore(Phi2toF')';
	Phi3toF = zscore(Phi3toF')';

    T = size(X3_Test);
    T = T(2)/2;
    Phi3TtoF = ((Phi3TtoF' - repmat(m3,2*T,1))./ repmat(s3,2*T,1))';
    
	%GT: Size = no. of examples x no. of features
	Z1 = Phi1toF';
	Z2 = Phi2toF';
	Z3 = Phi3toF';
    Z3_Test = Phi3TtoF';

	if save_data
		data_path = fullfile(filepath, out_filename);
		save(data_path,'Z1', 'Z2', 'Z3', 'Z3_Test', '-v6')

end
