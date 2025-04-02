#pragma once
#include <Eigen/Dense>

using namespace Eigen;

class QPSolver {

public:

    // Struct to perform Modified cholesky check
    struct ModChol {

        int w;
        MatrixXd L;
        MatrixXd D;
        MatrixXd C;
        bool flag;
        QPSolver* QPSolver;

        // operator for performing cholesky check
        MatrixXd operator()(MatrixXd& W);
    };

    int n; // Dimension of the problem
    int m; // Number of constraints

    // Inputs excluding solver_params
    VectorXd x; // Current state of the guess, nx1
    MatrixXd Q; // Quadratic cost (1/2x'*Q*x), nxn
    VectorXd F; // Linear cost (f'*x), nx1
    MatrixXd A; // Matrix associated with linear cost (Ax<=b) mxn
    VectorXd B; // Vector associated woth linear cost (Ax<=b) mx1

    // Variables that will be determined
    MatrixXd lambda; // Lagrange multiplier
    MatrixXd tslack; // Slack variables
    MatrixXd dx; // x's direction for step
    MatrixXd dlam; // lambda's direction for step
    MatrixXd dtsl; // tslack's direction for step
    float mu; // complimentary gap
    MatrixXd fval;
    int iter;

    //Tuneable variables
    int maxiter;	//between 5 and 1000
    float tol;		//between 1e-12 and 1e0
    float sigma;	//Convergence variable

    //Random params
    VectorXd  alg_diag;
    float zero_tol;
    float delta_l;
    float beta_l;
    float delta_u;
    float beta_u;
    float alpha_dec;

    // QPSolver's struct for modified cholesky check
    ModChol chol;

    // Class Definition
    QPSolver(MatrixXd& x_, MatrixXd& Q, MatrixXd& F, MatrixXd& A_, MatrixXd& B_, float solver_param[]);

    // Resets all determined and input variables
    void Reset(MatrixXd& x_, MatrixXd& Q_, MatrixXd& F_, MatrixXd& A_, MatrixXd& B_);

    // 0 means success, 1 means numerical issue, 2 means invalid input
    int ModMat(MatrixXd& M);

    //update's the optimal direction for all step direction variables
    void StepDir();

    // Main functionality method of QPSolver, iteratively
    void Solve();

};
