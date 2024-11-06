#include "QPS.h"

using namespace std;

class QPSolver {

public:

	int n;
	int m;

	MatrixXd x;

	MatrixXd Q;
	MatrixXd F;
	MatrixXd A;
	MatrixXd B;

	MatrixXd lambda;
	MatrixXd tslack;
	MatrixXd dx;
	MatrixXd dlam;
	MatrixXd dtsl;

	float mu;
	float fval;
	int iter;

	//Tuneable		May be assigned via readfile
	int maxiter = 10;	//between 5 and 1000
	float tol = 1e-6;		//between 1e-12 and 1e0
	float sigma = 0.6;	//Convergence var

	//Random params		????May be assigned via readfile for tests???
	MatrixXd alg_diag = MatrixXd::Zero(10,1);
	float zero_tol = 1e-16;
	float delta_l = 1e-16;
	float beta_l = 1e-16;
	float delta_u = 1e16;
	float beta_u = 1e16;
	float alpha_dec = 0.99;

	struct ModChol {

		MatrixXd L;
		MatrixXd D;
		bool flag;
		QPSolver* QPSolver;

		ModChol();

		~ModChol() {
			delete QPSolver;
		}

		//Function to Perform Modified cholesky check
		MatrixXd operator()(MatrixXd& M) {

			int m = M.rows();

			L = MatrixXd::Identity(m, m);

			D = MatrixXd::Zero(m, 1);

			MatrixXd C = MatrixXd::Zero(m, m);

			flag = false;


			for (int i = 0; i < m; i++) {

				float sum = 0;

				for (int j = 0; j < i - 1; j++) {

					sum += D(j) * L(i, j) * L(i, j);
				}

				C(i, i) = W(i, i) - sum;

				float cmax = 0;
				float cmin = QPSolver->beta_u;
				
				for (int j = i; j < m; j++) {
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

				//Determine d if too large
				//below is commented out on OG matlab code, need to ask
				if (d > (cmin / QPSolver->beta_u) * (cmin / QPSolver->beta_u)) {
					d = (cmax / QPSolver->beta_u) * (cmax / QPSolver->beta_u);
					flag = true;
				}

				if (d > QPSolver->delta_u) {
					d = QPSolver->delta_u;
					flag = true;
				}
				//above is commented out on OG matlab code, need to ask

				D(i) = d;

				for (int j = i; j < m; j++) {
					L(j, i) = C(j, i) / D(i);
				}
			}

			for (int i = 0; i < m; i++) {
				C.col(i) = L.col(i) * D(i);
			}

			return C * L.transpose();
		}

	};

	ModChol chol = ModChol();

	QPSolver();

	QPSolver(MatrixXd& x_, MatrixXd& Q_, MatrixXd& F_, MatrixXd& A_, MatrixXd& B_, float& solver_param[]) {
		n = max(x_.rows(), x_.cols()); //need to ask about this
		m = max(B_.rows(), B_.cols()); //need to ask
		x = x_;
		Q = Q_;
		F = F_;
		A = A_;
		B = B_;

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
		dx = MatrixXd::Zero(m, 1);
		dlam = MatrixXd::Zero(m, 1);
		dtsl = MatrixXd::Zero(m, 1);
		mu = (tslack.transpose() * lambda).value() / n;
		fval = 0.5 * (x.transpose() * Q * x).value() + (F.transpose() * x).value();
	}

	~QPSolver();

	void Reset(MatrixXd& x_, MatrixXd& Q_, MatrixXd& F_, MatrixXd& A_, MatrixXd& B_) {
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
		fval = 0.5 * (x.transpose() * Q * x).value() + (F.transpose() * x).value();
		alg_diag = MatrixXd::Zero(10, 1);
	}

	void StepDir() {
		MatrixXd rc = Q * x + A.transpose() * lambda + F;
		MatrixXd rg = -A * x - tslack + B;

		MatrixXd gamma1 = MatrixXd::Zero(m, 1);
		MatrixXd GTgamma = MatrixXd::Zero(n, m);

		for (int i = 0; i < m; i++) {
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

		if (alg_diag(0) == 0) {
			MatrixXd LHS = Q + GTgamma * A;
			MatrixXd RHS = -rc + A.transpose() * gamma1;
			dx = LHS.lu().solve(RHS);
			dlam = -gamma1;

			for (int i = 0; i < m; i++) {
				dlam(i) += (A.row(i) * dx * lambda(i) / tslack(i)).value();
				dtsl(i) = -tslack(i) + (sigma * mu - dlam(i)) * tslack(i) / lambda(i);
			}
		}
	}

	int ModMat(MatrixXd& M) {
		LLT<MatrixXd> llt(M);
		return llt.info();
	}

	void Solve() {
		float val3 = 1e10;

		for (int i = 0; i < maxiter; i++) {
			StepDir();
			float alpha = 1;

			while (true) {
				MatrixXd ltrial = lambda + dlam * alpha;
				MatrixXd ttrial = tslack + dtsl * alpha;

				float trialmin = 1e16;
				for (int j = 0; j < m; j++) {
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
		fval = 0.5 * (x.transpose() * Q * x).value() + (F.transpose() * x).value();
	}
};


