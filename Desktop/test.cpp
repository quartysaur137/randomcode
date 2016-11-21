#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <stdio.h>
#include <gsl/gsl_linalg.h>

// gcc test.cpp -lgsl -lgslcblas -o matrix
// ./matrix

int main (int argc, char** argv)
{
	int i=0;
	double x=atof(argv[1]);
	double y=atof(argv[2]);
	double z=atof(argv[3]);
	double pzz = 0;
	double pzo = 0;
	double poz = 0;
	double poo = 0;
	double sensordepth = 0;
	double sensorvelocity = 0;

	while ( i < 10 ) {
		double depth = x; // previous depth
		double velocity = y; // previous velocity
		double dt = 0.001; // delta time
		double accel = z; // previous accel?
		double otheraccel = z; //inputs from ahrs
		sensordepth = sensordepth + otheraccel * 0.5 * dt * dt;
		sensorvelocity = sensorvelocity + otheraccel * dt; // attempting to create some correlation between sensor readings and actual velocity/distance and not make them disconnected (as it's accelerometer/gyroscopic readings)
		int s;


		//whole bunch of array definitions; we then have a whole bunch of matrix definitions
		double identity1[] = {1};
		double identity2[] = {1, 0,
		                      0, 1 };
		double xk[] = { depth, 
		                velocity };
		double fk[] = { 1, dt, 
		                0, 1 };
		double bk[] = {dt*dt*accel*0.5,
		                dt*accel };
		double qk[] = {0.00000001032256, 0.000001032256,
                       0.000001032256,   0.0001032256   }; // noise from environment
		double pk[] = {pzz, pzo,
                       poz, poo};
		double rk[] = {1, 0,
		               0, -1}; // noise from observations? not supposed to be defined explicitly i think but in this case we're just substituting some random values in here
		double zk[] = {sensordepth, 
		               sensorvelocity };
		double hk[] = {0.5, 0,
		               0,   1 }; 
		double inva[4];



		gsl_matrix_view Fk = gsl_matrix_view_array(fk, 2, 2);
		gsl_matrix_view Xk = gsl_matrix_view_array(xk, 2, 1);
		gsl_matrix_view Bk = gsl_matrix_view_array(bk, 2, 1);
		gsl_matrix_view Iz = gsl_matrix_view_array(identity1, 1, 1);
		gsl_matrix_view Io = gsl_matrix_view_array(identity2, 2, 2);
		gsl_matrix_view Qk = gsl_matrix_view_array(qk, 2, 2);
		gsl_matrix_view Pk = gsl_matrix_view_array(pk, 2, 2);
		gsl_matrix_view Rk = gsl_matrix_view_array(rk, 2, 2);
		gsl_matrix_view Zk = gsl_matrix_view_array(zk, 2, 1);
		gsl_matrix_view Hk = gsl_matrix_view_array(hk, 2, 2);
		gsl_matrix_view inv = gsl_matrix_view_array(inva, 2, 2);
		gsl_matrix * xkk = gsl_matrix_calloc (2,1);
		gsl_matrix * qkk = gsl_matrix_calloc (2,2);
		gsl_matrix * qkkk = gsl_matrix_calloc (2,1);
		gsl_matrix * rkk = gsl_matrix_calloc (2,2);
		gsl_matrix * rkkk = gsl_matrix_calloc (2,1);
		gsl_matrix * pkk = gsl_matrix_calloc (2,2);
		gsl_matrix * Kk = gsl_matrix_calloc (2,2);
		gsl_matrix * Xkk = gsl_matrix_calloc (2,1);
		gsl_matrix * Pkk = gsl_matrix_calloc (2,2);
		gsl_permutation * p = gsl_permutation_alloc (2);


		//matrix multiplication stuff for kalman filter calculations.
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Fk.matrix, &Xk.matrix, 0.0, xkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Bk.matrix, &Iz.matrix, 1.0, xkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Fk.matrix, &Pk.matrix, 0.0, qkk);
		gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1.0, qkk, &Fk.matrix, 0.0, pkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Qk.matrix, &Io.matrix, 1.0, pkk); // end predict step


		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Hk.matrix, pkk, 0.0, qkk);
		gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1.0, qkk, &Hk.matrix, 0.0, rkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Rk.matrix, &Io.matrix, 1.0, rkk);
		gsl_linalg_LU_decomp (rkk, p, &s);
		gsl_linalg_LU_invert (rkk, p, &inv.matrix);
		gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1.0, pkk, &Hk.matrix, 0.0, qkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, qkk, &inv.matrix, 0.0, Kk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Hk.matrix, xkk, 0.0, rkkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &Zk.matrix, &Iz.matrix, -1.0, rkkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, Kk, rkkk, 0.0, qkkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, xkk, &Iz.matrix, 1.0, qkkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, qkkk, &Iz.matrix, 0.0, Xkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, Kk, &Hk.matrix, 0.0, qkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, qkk, pkk, 0.0, rkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, pkk, &Io.matrix, -1.0, rkk);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, rkk, &Io.matrix, 0.0, Pkk); // end Update step


		//get position/time stuff, covariances, and kalman gain stuff
		x = gsl_matrix_get (Xkk, 0, 0);
		y = gsl_matrix_get (Xkk, 1, 0);
		pzz = gsl_matrix_get (Pkk, 0, 0);
		pzo = gsl_matrix_get (Pkk, 0, 1);
		poz = gsl_matrix_get (Pkk, 1, 0);
		poo = gsl_matrix_get (Pkk, 1, 1);
		double kzz = gsl_matrix_get (Kk, 0, 0);
		double kzo = gsl_matrix_get (Kk, 0, 1);
		double koz = gsl_matrix_get (Kk, 1, 0);
		double koo = gsl_matrix_get (Kk, 1, 1);


		printf ("[%f\n",x);
		printf ("%f]\n",y);
		printf ("[%f, %f\n", pzz, pzo);
		printf ("%f, %f]\n", poz, poo);
		printf ("[%g, %g\n", kzz, kzo);
		printf ("%g, %g]\n", koz, koo);
		i++;
	}
	return 0;
}
