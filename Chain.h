/* This file is part of ESS-OO.
 *      Copyright (c) Marc Chadeau-Hyam (m.chadeau@imperial.ac.uk)
 *                    Leonardo Bottolo (l.bottolo@imperial.ac.uk)
 *                    David Hastie (d.hastie@imperial.ac.uk)
 *      2014
 *
 * Software designed and restructured by Loizos Markides (lm1011@ic.ac.uk)
 * ESS-OO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESS-OO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ESS-OO.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CHAIN_H_
#define CHAIN_H_
#include <memory>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

#include "Definitions.h"
#include "Score.h"
#include "ScoreESS.h"
#include "../UtilityClasses/Matrix.h"

using namespace std;

class Chain {
public:
	Chain();
	Chain(vector<unsigned int> newGammas);
	virtual ~Chain();
	void initialize(unsigned int nSweeps,unsigned int id, unsigned int resumeSweep);
	void newSweep();

	//LocalMoves
	void Gibbs(unsigned int nConfounders,
			  	  	  	 unsigned int maxPX,
			  	  	  	 unsigned int &count_0_1,
			  	  	 	 unsigned int &count_1_0,
			  	  		unsigned int &count_unchanged,
			  	  	  	 double g,
			  	  	  	 gsl_permutation* MyPerm,
			  	  	  	 gsl_rng* RandomNumberGenerator,
						  gsl_matrix* mat_Y,
						  double& prior_k);
	unsigned int FSMH(unsigned int nConfounders,
					  	unsigned int maxPX,
						unsigned int &count_nb_model_01,
						unsigned int &count_nb_accept_01,
						unsigned int &count_nb_model_10,
						unsigned int &count_nb_accept_10,
					  double g,
					  gsl_permutation* MyPerm,
					  gsl_rng* RandomNumberGenerator,
					  gsl_matrix* mat_Y,
					  double& prior_k);

	//Gamma vector methods
	void updateGammaVector(vector<unsigned int> newGammas);
	void reduceGammaVector(unsigned int *toRemove);
	void revertGamma(unsigned int var);
	void updateListOfIncludedVariables(vector<unsigned int> list_columns_X_gam);
	void getGammaVector(vector<unsigned int> *gam);

	//Score methods
	void calculateScore(double g,gsl_matrix* mat_Y,double& k_prior);

	//Getters and setters methods
	vector<unsigned int> getListOfIncludedVariables();
	vector<unsigned int> getGammaVector();
	unsigned int getGammaForVariable(unsigned int var);
	unsigned int getCurrentGammaVectorSize();
	unsigned int getVariablesIncluded();

	double getLogMarginalForSweep(unsigned int sweep);
	double getLogCondPostForSweep(unsigned int sweep);
	double getCurrentTemperature();
	string getGammaString();
	string getIndicesString();

	void getProposedScore(gsl_vector *prop_log_marg_condPost,
							   vector < unsigned int > vect_gam_prop,
							   double g,
							   gsl_matrix* mat_Y,
							   double k_prior);

	void setLogMarginalForSweep(double logMarginal,unsigned int sweep);
	void setLogCondPostForSweep(double logCondPost,unsigned int sweep);
	void setCurrentTemperature(double currentTemp);
	virtual void update_omega_k(double new_omega_k);
	virtual void update_rho_j(vector<double> &Rho_j);
	Score* score_;
	vector<unsigned int> gammas_;
protected:

	string log_;

	vector<unsigned int> list_columns_X_gam_;
	vector <double> logMarginal_;
	vector <double> logCondPost_;
	unsigned int current_sweep_;
	unsigned int n_vars_in_;
	unsigned int ID_;
	unsigned int resumeSweep_;

	double currentTemperature_;

	void updateListOfIncludedVariables();
	virtual double FSMH_updateTheta(unsigned int current_variable);
};

//Chain Object Pointers Containers
typedef std::shared_ptr<Chain> ChainPtr;
typedef std::vector<ChainPtr> ChainVectorPtr;
typedef ChainVectorPtr::iterator ChainIt;

#endif /* CHAIN_H_ */
