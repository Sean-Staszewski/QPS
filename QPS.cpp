#include "QPSolver.h"
#include <iostream>

MatrixXd QPSolver::ModChol::operator()(MatrixXd& W) {

    // Initializing
    w = W.rows();

    L = MatrixXd::Identity(w, w);
    D = MatrixXd::Zero(w, 1);
    C = MatrixXd::Zero(w, w);
    
    flag = false;

    for (int i = 0; i < w; i++) {

        float sum = 0;

        for (int j = 0; j < i - 1; j++) {

            sum += D(j) * L(i, j) * L(i, j);
        }

        C(i, i) = W(i, i) - sum;

        float cmax = 0;
        float cmin = QPSolver->beta_u;

        for (int j = i + 1; j < w; j++) {
            sum = 0;

            for (int k = 0; k < i - 1; k++) {
                sum = sum + D(k) * L(i, k) * L(j, k);
            }

            C(j, i) = W(j, i) - sum;

            if (abs(C(j, i)) > cmax) {
                cmax = abs(C(j, i));
            }
            if (abs(C(j, i)) < cmin) {
                cmin = abs(C(j, i));
            }
        }

        //Determine d if too small
        float d = abs(C(i, i));

        if (d < (cmax / QPSolver->beta_l) * (cmax / QPSolver->beta_l)) {
            d = (cmax / QPSolver->beta_l) * (cmax / QPSolver->beta_l);
            flag = true;
        }

        if (d < QPSolver->delta_l) {
            d = QPSolver->delta_l;
            flag = true;
        }

        //below is commented out on original matlab code

        //Determine d if too large
        //if (d > (cmin / QPSolver->beta_u) * (cmin / QPSolver->beta_u)) {
        //    d = (cmax / QPSolver->beta_u) * (cmax / QPSolver->beta_u);
        //    flag = true;
        //}

        //if (d > QPSolver->delta_u) {
        //    d = QPSolver->delta_u;
        //    flag = true;
        //}

        //above is commented out on original matlab code

        D(i) = d;

        for (int j = i + 1; j < w; j++) {
            L(j, i) = C(j, i) / D(i);
        }
    }

    for (int i = 0; i < w; i++) {
        C.col(i) = L.col(i) * D(i);
    }

    return C * L.transpose();
}

QPSolver::QPSolver(MatrixXd &x_, MatrixXd &Q_, MatrixXd &F_, MatrixXd &A_, MatrixXd &B_, float solver_param[]) {
    n = x_.rows();
    m = B_.rows();
    x = x_;
    Q = Q_;
    F = F_;
    A = A_;
    B = B_;

    //Tuneable
    maxiter = 10;	//between 5 and 1000
    tol = 1e-6;		//between 1e-12 and 1e0
    sigma = 0.6;	//Convergence var

    //Random params
    alg_diag = VectorXd::Zero(10);
    zero_tol = 1e-16;
    delta_l = 1e-16;
    beta_l = 1e-16;
    delta_u = 1e16;
    beta_u = 1e16;
    alpha_dec = 0.99;

    chol = ModChol();

    chol.QPSolver = this;

    if ((solver_param[0] >= 5) && (solver_param[0] <= 1000)) {
        maxiter = solver_param[0];
    }
    else {
        alg_diag(1) = 1;
    }
    if ((solver_param[1] > 1e-12) && (solver_param[1] < 1e0)) {
        tol = solver_param[1];
    }
    else {
        alg_diag(1) = 1;
    }

    if ((solver_param[2] > 10e-3) && (solver_param[2] < .8)) {
        sigma = solver_param[2];
    }
    else {
        alg_diag(1) = 1;
    }

    lambda = MatrixXd::Ones(m, 1);
    tslack = MatrixXd::Ones(m, 1);
    dx = MatrixXd::Zero(n, 1);
    dlam = MatrixXd::Zero(m, 1);
    dtsl = MatrixXd::Zero(m, 1);
    mu = (tslack.transpose() * lambda).value() / n;
    fval = 0.5 * (x.transpose() * Q * x) + (F.transpose() * x);

    std::cout << "alg_diag after initialization: " << alg_diag.size() << std::endl;
}

void QPSolver::Reset(MatrixXd& x_, MatrixXd& Q_, MatrixXd& F_, MatrixXd& A_, MatrixXd& B_) {
    x = x_;
    Q = Q_;
    F = F_;
    A = A_;
    B = B_;

    lambda = MatrixXd::Ones(m, 1);
    tslack = MatrixXd::Ones(m, 1);
    dx = MatrixXd::Zero(m, 1);
    dlam = MatrixXd::Zero(m, 1);
    dtsl = MatrixXd::Zero(m, 1);
    mu = (tslack.transpose() * lambda).value() / n;
    fval = 0.5 * (x.transpose() * Q * x) + (F.transpose() * x);
    alg_diag = MatrixXd::Zero(10, 1);
}

int QPSolver::ModMat(MatrixXd &M) {
    LLT<MatrixXd> llt(M);
    if (llt.info() == Eigen::Success) {
        return 0;
    }
    else if (llt.info() == Eigen::NumericalIssue) {
        return 1;
    }
    else {
        return 2;
    }
}

void QPSolver::StepDir() {

    std::cout << "alg_diag in StepDir: " << alg_diag.size() << std::endl;

    MatrixXd rc = Q * x + A.transpose() * lambda + F; //getting an error here indicating dimensions don't work for the math, suspecting its lambda because it is a vector
    MatrixXd rg = -A * x - tslack + B;

    MatrixXd gamma1 = MatrixXd::Zero(m, 1);
    MatrixXd GTgamma = MatrixXd::Zero(n, m);

    for (int i = 0; i < m; i++) {
        std::cout << "in first loop StepDir" << std::endl;
        if (tslack(i) < zero_tol || lambda(i) <= zero_tol) {
            alg_diag(0) = -1;
            alg_diag(2) = -1;
            break;
        }
        else {
            gamma1(i) = rg(i) * lambda(i) / tslack(i) + lambda(i) - sigma * mu / tslack(i);
            GTgamma.col(i) = A.row(i).transpose() * lambda(i) / tslack(i);
        }
    }
    std::cout << "passed first loop StepDir" << std::endl;

    if (alg_diag(0) == 0) {
        std::cout << "made if alg_diag check StepDir" << std::endl;
        MatrixXd LHS = Q + GTgamma * A;
        MatrixXd RHS = -rc + A.transpose() * gamma1;
        dx = LHS.lu().solve(RHS);
        dlam = -gamma1;

        //combine forloops for runtime????????
        for (int i = 0; i < m; i++) {
            std::cout << "got to loop1 StepDir" << std::endl;
            dlam(i) = dlam(i) + (A.row(i) * dx * lambda(i) / tslack(i)).value();
        }
        for (int i = 0; i < m; i++) {
            std::cout << "got to loop2 StepDir" << std::endl;
            dtsl(i) = -tslack(i) + (sigma * mu - dlam(i)) * tslack(i) / lambda(i);
        }
    }
    std::cout << "made it out alive" << std::endl;
}

void QPSolver::Solve() {
    std::cout << "alg_diag in Solve: " << alg_diag.size() << std::endl;

    float val3 = 1e10;

    for (int i = 0; i < maxiter; i++) {
        //std::cout << i << std::endl;
        this->StepDir();
        float alpha = 1;

        while (true) {
            MatrixXd ltrial = lambda + dlam * alpha;
            MatrixXd ttrial = tslack + dtsl * alpha;

            float trialmin = 1e16;

            for (int j = 0; j < this->m; j++) {
                if (ltrial(j) < trialmin) {
                    trialmin = ltrial(j);
                }
                if (ttrial(j) < trialmin) {
                    trialmin = ttrial(j);
                }
            }

            if (trialmin > 0) {
                break;
            }
            else {
                alpha *= alpha_dec;
            }

            if (alpha < zero_tol) {
                alg_diag(0) = 2;
                break;
            }
        }

        alpha *= 0.995;

        dx *= alpha;
        dlam *= alpha;
        dtsl *= alpha;

        x += dx;
        std::cout << x << std::endl;
        lambda += dlam;
        tslack += dtsl;

        float max_val = 0;

        for (int j = 0; j < n; j++) {
            if (abs(dx(j)) > max_val) {
                max_val = abs(dx(j));
            }
        }

        for (int j = 0; j < m; j++) {
            if (abs(dtsl(j)) > max_val) {
                max_val = abs(dtsl(j));
            }
        }

        if (max_val < tol) {
            alg_diag(0) = 1;
        }

        else {
            //Now check KKT cond
            MatrixXd val1 = Q * x + A.transpose() * lambda + F;
            MatrixXd val2 = -A * x - tslack + B;
            val3 = (tslack.transpose() * lambda).value();

            max_val = 0;

            for (int j = 0; j < n; j++) {
                if (abs(val1(j)) > max_val) {
                    max_val = abs(val1(j));
                }
            }

            for (int j = 0; j < m; j++) {
                if (abs(val2(j)) > max_val) {
                    max_val = abs(val2(j));
                }
            }

            if (val3 > max_val) {
                max_val = val3;
            }
            if (max_val < tol) {
                alg_diag(0) = 1;
            }
        }

        mu = val3 / m;

        if (alg_diag(0) != 0) {
            iter = i;
            break;
        }
    }
    fval = 0.5 * (x.transpose() * Q * x) + (F.transpose() * x);
}
