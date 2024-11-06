#pragma once
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

class QPSolver {

public:

	struct ModChol {

		MatrixXd L;
		MatrixXd D;
		int flag;
		QPSolver* QPSolver;

		ModChol();

		MatrixXd operator()(MatrixXd& M);

	};

	//Inputs:
	int n;
	int m;

	MatrixXd x;
	MatrixXd Q;
	MatrixXd F;
	MatrixXd A;
	MatrixXd B;

	//Determined vars
	MatrixXd lambda;
	MatrixXd tslack;
	MatrixXd dx;
	MatrixXd dlam;
	MatrixXd dtsl;

	float mu;
	float fval;
	int iter;

	//Tuneable vars
	int maxiter;	//between 5 and 1000
	float tol;		//between 1e-12 and 1e0
	float sigma;	//Convergence var

	//Random params
	MatrixXd alg_diag;
	float zero_tol;
	float delta_l;
	float beta_l;
	float delta_u;
	float beta_u;
	float alpha_dec;

	ModChol chol;

	QPSolver();

	QPSolver(MatrixXd& x_, MatrixXd& Q, MatrixXd& F, MatrixXd& A_, MatrixXd& B_, float& solver_param);

	~QPSolver();

	void Reset();

	void StepDir();

	MatrixXd ModMat(MatrixXd& M);

	void Solve();
};