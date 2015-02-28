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

#include "Chain.h"
#include "MCMC.h"
#include "../UtilityClasses/MatrixManipulation.h"
#define DEBUG 0


Chain::Chain() {
	score_ = new ScoreESS();
	n_vars_in_=0;
	current_sweep_=0;
	ID_=0;
	currentTemperature_=1.0;
	resumeSweep_ = 0;
}

Chain::Chain(vector<unsigned int> newGammas){
	current_sweep_=0;
	currentTemperature_=1.0;
	score_ = new ScoreESS();
	gammas_ = newGammas;
	updateListOfIncludedVariables();
}

Chain::~Chain() {
	gammas_.clear();
	logMarginal_.clear();
	logCondPost_.clear();
	list_columns_X_gam_.clear();
}

vector<unsigned int> Chain::getGammaVector()
{
	return gammas_;
}

void Chain::getGammaVector(vector<unsigned int> *gam)
{
	gam = &gammas_;
}

string Chain::getGammaString()
{
	//Get the gamma vector as a string
	std::ostringstream stringgammas;
	//std::copy(gammas_.begin(), gammas_.end(), std::ostream_iterator<bool>(std::cout, " "));
	std::copy(gammas_.begin(), gammas_.end(), std::ostream_iterator<bool>(stringgammas,""));
	return stringgammas.str();
}

string Chain::getIndicesString()
{
	//Get the indices as a string
	std::ostringstream stringgammas;
	//std::copy(gammas_.begin(), gammas_.end(), std::ostream_iterator<bool>(std::cout, " "));
	std::copy(list_columns_X_gam_.begin(), list_columns_X_gam_.end(), std::ostream_iterator<unsigned int>(stringgammas,","));
	return stringgammas.str();
}

void Chain::revertGamma(unsigned int var)
{
	gammas_[var] = 1 - gammas_[var];
	updateListOfIncludedVariables();
}

void Chain::initialize(unsigned int nSweeps,unsigned int id, unsigned int resumeSweep)
{
	logMarginal_.resize(nSweeps);
	logCondPost_.resize(nSweeps);
	ID_ = id;
	resumeSweep_ = resumeSweep;
}

void Chain::updateGammaVector(vector<unsigned int> newGammas)
{
	// This function first updates the gammas of a function
	// and then changes the list of included variables
	gammas_ = newGammas;
	updateListOfIncludedVariables();
}

void Chain::updateListOfIncludedVariables(vector<unsigned int> list_columns_X_gam)
{
	// overloaded function that first updates the list of included variables
	// and then changes the gammas of a function
	list_columns_X_gam_.clear();
	list_columns_X_gam_ = list_columns_X_gam;

	unsigned int gammas_size = MCMC::pX;

	gammas_ = vector<unsigned int>(gammas_size,0);
	n_vars_in_=list_columns_X_gam_.size();

	for (unsigned int i=0;i<n_vars_in_;i++)
	{
		gammas_[list_columns_X_gam_[i]] = 1;
	}
}

void Chain::updateListOfIncludedVariables()
{
	list_columns_X_gam_.clear();
	MatrixManipulation::get_list_var_in(list_columns_X_gam_,
						  	  	  	  	  	gammas_);

	n_vars_in_=list_columns_X_gam_.size();
}

vector<unsigned int> Chain::getListOfIncludedVariables()
{
	return list_columns_X_gam_;
}

void Chain::calculateScore(double g,gsl_matrix* mat_Y,double& k_prior)
{
	double logMargLik,logPost;

	score_->computeLogPosterior(logMargLik,
	                    logPost,
	                    list_columns_X_gam_,
	                    g,mat_Y, k_prior,MCMC::pX);

	logMarginal_[current_sweep_]=logMargLik;
	logCondPost_[current_sweep_]=logPost;
}

void Chain::newSweep()
{
	++current_sweep_;
	logMarginal_[current_sweep_] = logMarginal_[current_sweep_-1];
	logCondPost_[current_sweep_] = logCondPost_[current_sweep_-1];
}



unsigned int Chain::getCurrentGammaVectorSize()
{
	return gammas_.size();
}

unsigned int Chain::getVariablesIncluded()
{
	return n_vars_in_;
}

double Chain::getLogMarginalForSweep(unsigned int sweep)
{
	return logMarginal_[sweep-resumeSweep_];
}
double Chain::getLogCondPostForSweep(unsigned int sweep)
{
	return logCondPost_[sweep-resumeSweep_];
}
void Chain::setLogMarginalForSweep(double logMarginal,unsigned int sweep)
{
	logMarginal_[sweep-resumeSweep_] = logMarginal;
}

void Chain::setLogCondPostForSweep(double logCondPost,unsigned int sweep)
{
	logCondPost_[sweep-resumeSweep_]=logCondPost;
}

unsigned int Chain::getGammaForVariable(unsigned int var)
{
	return gammas_[var];
}

void Chain::setCurrentTemperature(double currentTemp)
{
	currentTemperature_ = currentTemp;
}

double Chain::getCurrentTemperature()
{
	return currentTemperature_;
}

void Chain::reduceGammaVector(unsigned int *toRemove)
{
	unsigned int j=0,k=0;
	for(unsigned int col=0;col<getCurrentGammaVectorSize();col++)
	{
		if(gammas_[col]==1)
		{
			if(j==toRemove[k])
			{
				gammas_[col]=0;
				++k;
			}
			++j;
		}
	}

	updateListOfIncludedVariables();
}

unsigned int Chain::FSMH(unsigned int nConfounders,
				  unsigned int maxPX,
					unsigned int &count_nb_model_01,
					unsigned int &count_nb_accept_01,
					unsigned int &count_nb_model_10,
					unsigned int &count_nb_accept_10,
				  double g,
				  gsl_permutation* MyPerm,
				  gsl_rng* RandomNumberGenerator,
				  gsl_matrix* mat_Y,
				  double& prior_k)
{


	//This function runs the FSMH move for the specific chain that calls it.
	double reverseTemperature = 1.0/currentTemperature_;

	//chain specific information
	double temp_log_marg=logMarginal_[current_sweep_];
	double temp_log_cond_post=logCondPost_[current_sweep_];
	double proposed_log_condPost=0;
	unsigned int sum_gam=n_vars_in_;
	unsigned int n_Models_Visited=0;

	//Step1: get the random order of variable: shuffle the permutation object
	gsl_ran_shuffle (RandomNumberGenerator, MyPerm->data, MyPerm->size, sizeof(size_t));

	for(unsigned int current_variable=0;current_variable<MyPerm->size;current_variable++)
	{
		unsigned int pos_curr_var=MyPerm->data[current_variable];

		if(pos_curr_var<nConfounders)
		{
			continue;
		}

		unsigned int current_value=getGammaForVariable(pos_curr_var);
		double rand_test=gsl_rng_uniform(RandomNumberGenerator);
		double alpha_FSMH=0.0;
		bool doMove =false;

		//Step 2: Calculate alpha_FSMH: probability to make a move
		bool autoReject = false;
		if(current_value==0)
		{
			// Note in the next line, because p_gamma (sum gam) is the value with the
			// current covariate set to 0, we need to add 1 to the numerator
			double theta_1= FSMH_updateTheta(current_variable);

			double theta_0=1.0-theta_1;
			double theta_1_tilda=theta_1;
			double theta_0_tilda=theta_0;

			//Step 2.1: in the variable is not in p(alpha_FSMH=0.0)=theta_1_tilda
			if(rand_test<theta_1_tilda)
			{
				doMove = true;
				//Try to change gamma 0->1

				count_nb_model_01++;

				//double log_L=current_log_marg*(1.0/(*t_tun).t[current_chain]);
				double log_P=temp_log_cond_post*reverseTemperature;

				//Step 2.2: if alpha_FSMH!=0.0, alpha_FSMH=f(log_prob,theta_1...)
				revertGamma(pos_curr_var);

				//Getting X_gamma
				if(n_vars_in_>maxPX)
				{
					// selected variables are more than the max accepted value
					autoReject=true;
					alpha_FSMH=0.0;
				}
				else
				{


					calculateScore(g, mat_Y,prior_k);

					//Calculate log_cond_post for the proposed move

					proposed_log_condPost=logCondPost_[current_sweep_];

					n_Models_Visited++;

					double temp=log(theta_0_tilda)-log(theta_1_tilda)+proposed_log_condPost*reverseTemperature-log_P;
					alpha_FSMH=min(1.0, exp(temp));
				}
			}
			else
			{//Nothing to do
			}
		}
		else
		{//current_value==1
			double theta_1= FSMH_updateTheta(current_variable);

			double theta_0=1.0-theta_1;
			double theta_1_tilda=theta_1;
			double theta_0_tilda=theta_0;
			//Step 2.1: in the variable is not in p(alpha_FSMH=0.0)=theta_0_tilda

			if(rand_test<theta_0_tilda)
			{
				doMove = true;

				count_nb_model_10++;

						double log_P=temp_log_cond_post*reverseTemperature;

						//Step 2.2: if alpha_FSMH!=0.0, alpha_FSMH=f(log_prob,theta_1...)
						revertGamma(pos_curr_var);

						//Calculate log_cond_post for the proposed move
						calculateScore(g, mat_Y,prior_k);

						proposed_log_condPost=logCondPost_[current_sweep_];

						n_Models_Visited++;

						double temp=log(theta_1_tilda)-log(theta_0_tilda)+proposed_log_condPost*reverseTemperature-log_P;
						alpha_FSMH=min(1.0, exp(temp));
					}
					else
					{//Nothing to do
					}
				}//end of current variable==1

				if(doMove)
				{
					rand_test = gsl_rng_uniform(RandomNumberGenerator);

					if(rand_test >= alpha_FSMH||autoReject)
					{//Move rejected
						revertGamma(pos_curr_var);
						logCondPost_[current_sweep_] = temp_log_cond_post;
						logMarginal_[current_sweep_] = temp_log_marg;
					}
					else
					{
						if(getGammaForVariable(pos_curr_var)==1)
						{//Move 0->1 accepted
							sum_gam++;
							count_nb_accept_01++;
						}
						else
						{//Move 1->0 accepted
							count_nb_accept_10++;
							sum_gam--;
						}

						temp_log_cond_post=proposed_log_condPost;
						temp_log_marg=logMarginal_[current_sweep_];
					}
				}
	}
	return n_Models_Visited;
}

double Chain::FSMH_updateTheta(unsigned int current_variable)
{
	double alpha=Prior::w[1]*Prior::w[0];
	double beta=Prior::w[1]*(1.0-Prior::w[0]);

	return max(0.0, (double)(n_vars_in_+alpha - getGammaForVariable(current_variable))
						/ (double)(MCMC::pX+alpha+beta-1));
}

void Chain::Gibbs(unsigned int nConfounders,
		  	  	  	 unsigned int maxPX,
		  	  		unsigned int &count_0_1,
		  	  		unsigned int &count_1_0,
		  	  		unsigned int &count_unchanged,
		  	  	  	 double g,
		  	  	  	 gsl_permutation* MyPerm,
		  	  	  	 gsl_rng* RandomNumberGenerator,
					  gsl_matrix* mat_Y,
					  double& prior_k)
{
	// This function runs Gibbs Move on the specific chain that calls it.

	//Step1: get the random order of variable: shuffle the permutation object
	gsl_ran_shuffle (RandomNumberGenerator, MyPerm->data, MyPerm->size, sizeof(size_t));

	double proposed_log_cond_post=0.0;
	double current_log_cond_post=logCondPost_[current_sweep_];
	double current_log_marg=logMarginal_[current_sweep_];

	//Step 2: for each variable calculating the moving pbty
	for(unsigned int current_variable=0;current_variable<MyPerm->size;current_variable++)
	{
		bool scoreUpdated = false;
		//(*My_Move_monitor).Gibbs_nb_model++;
	    unsigned int pos_curr_var=MyPerm->data[current_variable];
	    if(pos_curr_var<nConfounders)
	    {
	      // confounders don't get updated as always in
	      continue;
	    }

		unsigned int current_value=getGammaForVariable(pos_curr_var);

		//Step 3: Calculate the alternative long_cond_post
		//Step 3.1: change Gamma, and get the new proposed X_gam
		revertGamma(pos_curr_var);
		//Initial Calculation of the logMarg and log_cond_post;
		//Step 1: Getting X_gamma
		double log_condPost0,log_condPost1;
		double argument,theta=0.0;
		if(n_vars_in_<=maxPX)
		{
			calculateScore(g, mat_Y,prior_k);
			scoreUpdated=true;
		    //Step 2: Calculate log_cond_post if move
		    proposed_log_cond_post=logCondPost_[current_sweep_];

		    //Step 3: Sample the new status
		    // We prevent n_var_in moving to more than maxPX later below
		    log_condPost0=0.0;
		    log_condPost1=0.0;

		    //The sampling theta = 1/(1+exp(condpost(0)-condpost(1)))
		    if(current_value==0)
		    {//i.e. 0->1 move
		    	log_condPost0=current_log_cond_post;
		    	log_condPost1=proposed_log_cond_post;
		    }
		    else
		    {//i.e. 1->0 move
		    	log_condPost1=current_log_cond_post;
		    	log_condPost0=proposed_log_cond_post;
		    }

		    argument=(log_condPost0-log_condPost1)/(currentTemperature_);
		    theta=1.0/(1.0+exp(argument));
		}

		unsigned int sampled_value;
		if(current_value==0&&n_vars_in_>maxPX)
		{
		 	// Stop from adding more variables than permitted (as this would have 0 prob)
		   	sampled_value=0;
		}
		else
		{
			// Otherwise perform the sample
		    sampled_value=(unsigned int)(gsl_ran_bernoulli(RandomNumberGenerator,theta));
		}
		    //Step 4: updating vectors
		    if(current_value==sampled_value)
		    {
		    	//Revert vect_gam
		    	revertGamma(pos_curr_var);
		    	if (scoreUpdated)
		    	{
		    		//Revert score
		    		logCondPost_[current_sweep_]=current_log_cond_post;
		    		logMarginal_[current_sweep_] = current_log_marg;
		    		proposed_log_cond_post=current_log_cond_post;
		    	}
		        count_unchanged++;
		    }
		    else
		    {
		    	//updating log_marg and log_condPost
		    	current_log_cond_post=logCondPost_[current_sweep_];
		    	current_log_marg=logMarginal_[current_sweep_];

		    	if(current_value==0)
		    	{
		    		count_0_1++;
		    	}
		    	else
		    	{
		    		count_1_0++;
		    	}
		    }
		}//end of for variable
}


void Chain::update_omega_k(double new_omega_k)
{
	//double diff =
	score_->update_omega_k(new_omega_k,list_columns_X_gam_,gammas_.size());
	//logCondPost_[current_sweep_] += diff;
}
void Chain::update_rho_j(vector<double> &Rho_j)
{
	//double diff =
	score_->update_rho_j(Rho_j,list_columns_X_gam_,gammas_.size());
	//logCondPost_[current_sweep_] += diff;
}


void Chain::getProposedScore(gsl_vector *prop_log_marg_condPost,
						   vector < unsigned int > vect_gam_prop,
						   double g,
						   gsl_matrix* mat_Y,
						   double k_prior)
{

	vector < unsigned int > old_vector;
	old_vector.resize(gammas_.size());

	copy(gammas_.begin(),gammas_.end(),old_vector.begin());

	updateGammaVector(vect_gam_prop);
	double currentLogMarg=getLogMarginalForSweep(current_sweep_);
	double currentLogCondPost=getLogCondPostForSweep(current_sweep_);

	calculateScore(g,mat_Y,k_prior);

	prop_log_marg_condPost->data[0]=getLogMarginalForSweep(current_sweep_);
	prop_log_marg_condPost->data[1]=getLogCondPostForSweep(current_sweep_);

	//revert
	setLogMarginalForSweep(currentLogMarg,current_sweep_);
	setLogCondPostForSweep(currentLogCondPost,current_sweep_);
	updateGammaVector(old_vector);
}
