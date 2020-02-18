/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2017 The ves-code team
   (see the PEOPLE-VES file at the root of this folder for a list of names)

   See http://www.ves-code.org for more information.

   This file is part of ves-code, version 1.

   ves-code is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ves-code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with ves-code.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "TargetDistribution.h"
#include "GridIntegrationWeights.h"
#include "VesBias.h"
#include "core/ActionRegister.h"
#include "tools/Grid.h"
#include <iostream>
#include "tools/FileBase.h"
#include "Optimizer.h"
#include "LinearBasisSetExpansion.h"
#include "VesLinearExpansion.h"
#include "tools/Communicator.h"

namespace PLMD {
namespace ves {

//+PLUMEDOC VES_TARGETDIST TD_TS_TARGET
/*
TS-target target distribution (dynamic).

This target distribution can be used to modulate the probability of sampling 
the different different stationary states of the Free Energy Surface.

The SCALEFACTOR determines the width of the peaks in the p(s). The optional
keyword LAMBDA (default: 0.1) determines how fast the switching function decays 
and the optional keyword LIMIT (default: 1) is used to specify the scaling
between the probabilities of sampling the minima and the maxima.

An optional keyword PRINT_STRIDE can be used to specify the frequency of 
printing the first derivative, second derivative and swith grids. In the absence 
of this keyword these quantities are never printed.

The target distribution is dynamic and so the frequency of updating the target 
needs to be set in the optimizer used in the calculation. 

NOTE: This target distribution works only for one-dimensional cases.

\par Examples
Employ a TS-target distribution to sample the stationary points of F(s)
with equal probability
\plumedfile
TD_TS_TARGET ...
  SCALEFACTOR=20
  LAMBDA=0.1    
  LIMIT=1.0
  PRINT_STRIDE=500
  LABEL=td
... TD_TS_TARGET
\endplumedfile

Employ a TS-target distribution to sample the stationary points of F(s)
such that the maxima are visited with a probability two times that of the minima
\plumedfile
TD_TS_TARGET ...
  SCALEFACTOR=20
  LAMBDA=0.1    
  LIMIT=0.5
  PRINT_STRIDE=500
  LABEL=td
... TD_TS_TARGET
\endplumedfile

*/
//+ENDPLUMEDOC

class TD_TS_TARGET: public TargetDistribution {
private:
  unsigned int iter;
  Grid* derv_grid_pntr_ ;
  Grid* derv2_grid_pntr_ ;
  Grid* switch_grid_pntr_ ;
  double scale_factor_;
  int pstride;
  void calculatefesCoeff();   // expresses the fes_grid as a linear expansion of basis functions and returns the coefficients
  void calculateFirstDerv();  // returns a grid of values beta*dF(s)/ds
  void calculateSecondDerv(); // returns a grid of values d(beta*dF(s)/ds)/ds
  double lambda;
  double limit;
  std::vector<unsigned int> nbasisf;
  unsigned int ncoeffs;
  std::vector<BasisFunctions*> pntrs_to_basisf_ ;
  LinearBasisSetExpansion* pntr_to_bias_expansion_;
  CoeffsVector* fes_coeffs_pntr_;
  size_t stride;
  size_t rank;
  std::vector<double> bf_norm;
  std::string derv_init_filename_;
  std::string derv2_init_filename_;
  std::string switch_init_filename_;
public:
  static void registerKeywords(Keywords&);
  explicit TD_TS_TARGET(const ActionOptions& ao);
  void updateGrid();
  double getValue(const std::vector<double>&) const;
  ~TD_TS_TARGET() {}
};


PLUMED_REGISTER_ACTION(TD_TS_TARGET,"TD_TS_TARGET")


void TD_TS_TARGET::registerKeywords(Keywords& keys) {
  TargetDistribution::registerKeywords(keys);
  keys.add("compulsory","SCALEFACTOR","The scale factor used for the Lorentzian distribution.");
  keys.add("optional","LAMBDA", "The lambda second derivative switch");
  keys.add("optional","LIMIT", "The lower limit of the switch");
  keys.add("optional","PRINT_STRIDE","The frequency of printing the gradient of the FES");
}


TD_TS_TARGET::TD_TS_TARGET(const ActionOptions& ao):
  PLUMED_VES_TARGETDISTRIBUTION_INIT(ao),
  iter(0),
  derv_grid_pntr_(NULL),
  derv2_grid_pntr_(NULL),
  switch_grid_pntr_(NULL),
  scale_factor_(0.0),
  pstride(0),
  lambda(0.1),
  limit(0.0),
  nbasisf(0.0),
  ncoeffs(0),
  pntrs_to_basisf_(0.0),
  pntr_to_bias_expansion_(NULL),
  fes_coeffs_pntr_(NULL),
  stride(1),
  rank(0),
  bf_norm(1.0)
{
  parse("SCALEFACTOR",scale_factor_);
  if(keywords.exists("LAMBDA")) {
    parse("LAMBDA",lambda);
  }
  if(keywords.exists("LIMIT")) {
    parse("LIMIT",limit);
  }
  if(keywords.exists("PRINT_STRIDE")) {
    parse("PRINT_STRIDE",pstride);
  }
  setDynamic();
  setFesGridNeeded();
  checkRead();

}


double TD_TS_TARGET::getValue(const std::vector<double>& argument) const {
  plumed_merror("getValue not implemented for TD_TS_TARGET");
  return 0.0;
}


void TD_TS_TARGET::updateGrid() {
//--------------- obtain the number of basis functions, type of basis function ---------------

  if(pntr_to_bias_expansion_ == NULL){
    VesBias* vesbias = getPntrToVesBias();
    VesLinearExpansion* veslinear = static_cast<VesLinearExpansion*>(vesbias);
    if(veslinear) {
      pntrs_to_basisf_ = veslinear->get_basisf_pntrs();
      pntr_to_bias_expansion_ = veslinear->get_bias_expansion_pntr(); 
    }
    else{
      plumed_merror("Use Ves_Linear_Expansion type for VesBias when the target distribution is Lorentzian");
    }
    nbasisf.assign(pntr_to_bias_expansion_->getNumberOfArguments(),0.0);
    ncoeffs=pntr_to_bias_expansion_->getNumberOfCoeffs();
    for(unsigned int i=0; i<pntr_to_bias_expansion_->getNumberOfArguments(); i++){
      nbasisf[i]=pntr_to_bias_expansion_->getNumberOfBasisFunctions()[i];
      if(pntrs_to_basisf_[i]->getType()!="trigonometric_cos-sin" and pntrs_to_basisf_[i]->getType()!="Legendre"){
        plumed_merror("Only BF_Fourier or BF_Legendre as of now");
      }
      if(pntrs_to_basisf_[i]->getType()=="trigonometric_cos-sin"){
        bf_norm[i]=(pntrs_to_basisf_[i]->intervalRange())/2.0;
      }
      else{
        bf_norm[i]=1.0;
      }
    }
    derv_init_filename_="FESderv1."+vesbias->getLabel()+".";
    derv2_init_filename_="FESderv2."+vesbias->getLabel()+".";
    switch_init_filename_="switch."+vesbias->getLabel()+".";
  }
 
  Communicator& comm_in = pntr_to_bias_expansion_->getPntrToBiasCoeffs()->getCommunicator();
  stride=comm_in.Get_size();
  rank=comm_in.Get_rank();

  double scale = scale_factor_ * scale_factor_;

  plumed_massert(getFesGridPntr()!=NULL,"the FES grid has to be linked to use TD_Lorentzian!");
 
// --------------- calculate the first and second derivatives of F(s) ---------------
  calculateFirstDerv();  
  calculateSecondDerv();  

// --------------- calculate the new target distribution ---------------
  std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getTargetDistGridPntr());
  double norm = 0.0;
  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
    double value = derv_grid_pntr_->getValue(l);
    value = value*value;
    value = scale + value;
    value = scale_factor_ /value;
    norm += integration_weights[l]*value;
    targetDistGrid().setValue(l,value);
  }
  targetDistGrid().scaleAllValuesAndDerivatives(1.0/norm);
  norm = 0.0;
    for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
      double value = targetDistGrid().getValue(l);
      value = value*switch_grid_pntr_->getValue(l);
      norm += integration_weights[l]*value;
      targetDistGrid().setValue(l,value);
      value = -std::log(value);
      logTargetDistGrid().setValue(l,value);
    }
    targetDistGrid().scaleAllValuesAndDerivatives(1.0/norm);
  logTargetDistGrid().setMinToZero();

}

void TD_TS_TARGET::calculatefesCoeff(){

// --------------- initializes quantities ---------------
  std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getFesGridPntr());
  std::vector<double> coeff;
  coeff.assign(ncoeffs,0.0);
  double norm=0.0;
  std::vector<double> product(nbasisf[0],0.0);

// --------------- calulates the fes coefficients by integrating over the grid points ---------------
  for(Grid::index_t l=0; l<getFesGridPntr()->getSize(); l++) {
    std::vector<double> args =(getFesGridPntr()->getPoint(l));
    double fes = getFesGridPntr()->getValue(l);
    fes= integration_weights[l]*(fes);
    unsigned int nargs = args.size();
    std::vector<double> args_values_trsfrm(nargs);
    std::vector< std::vector <double> > basisf(nargs);
    std::vector< std::vector <double> > derivs(nargs);
    std::vector< std::vector <double> > inner_pdt(nargs);
    bool all_inside;
    for(unsigned int k=0; k<nargs; k++) {
      basisf[k].assign(nbasisf[k],0.0);
      derivs[k].assign(nbasisf[k],0.0);
      pntrs_to_basisf_[k]->getAllValues(args[k],args_values_trsfrm[k],all_inside,basisf[k],derivs[k]);
    }
    if(l==0) {
      inner_pdt=(pntrs_to_basisf_[0]->getAllInnerProducts());
      for(size_t i=0; i<ncoeffs; i+=1){
        std::vector<unsigned int> indices=getFesGridPntr()->getIndices(i);
        for(unsigned int k=0; k<nargs; k++) {
          norm+=inner_pdt[k][indices[k]];
        }
      }
    }
    for(size_t i=0; i<ncoeffs; i+=1){
      std::vector<unsigned int> indices=getFesGridPntr()->getIndices(i);
      double basis_curr = 1.0;
      for(unsigned int k=0; k<nargs; k++) {
        basis_curr*= basisf[k][indices[k]]/bf_norm[k] ;
      }
      coeff[i]+=((fes)*basis_curr/norm);
    }
  }
    for(size_t i=0; i<ncoeffs; i+=1){
      fes_coeffs_pntr_->setValue(i,coeff[i]);
  }
}  

void TD_TS_TARGET::calculateFirstDerv(){
   Communicator& comm_in = pntr_to_bias_expansion_->getPntrToBiasCoeffs()->getCommunicator();

   iter= iter+1;

// --------------- initializes quantities at the first iteration ---------------
   if(derv_grid_pntr_==NULL){ 
     std::vector<double> dx_ = getFesGridPntr()->getDx();
     std::vector<unsigned> nbins = pntr_to_bias_expansion_->getGridBins(); //getFesGridPntr()->getNbin();
     std::vector<std::string> min = getFesGridPntr()->getMin();
     std::vector<std::string> max = getFesGridPntr()->getMax();
     std::vector<std::string> args = getFesGridPntr()->getArgNames();
     std::vector<bool> isperiodic = getFesGridPntr()->getIsPeriodic();
     derv_grid_pntr_ = new Grid("grad_fes",args,min,max,nbins,false,false,true,isperiodic,min,max);
   }
   if(fes_coeffs_pntr_ == NULL){
     std::vector<Value*> args_pntrs_= pntr_to_bias_expansion_->getPntrsToArguments();
     fes_coeffs_pntr_ = new CoeffsVector("fes.coeffs",args_pntrs_,pntrs_to_basisf_,comm_in,true);
     fes_coeffs_pntr_->setAllValuesToZero();
   }

// --------------- calculates the fes coeffs using the fes grid ---------------
   calculatefesCoeff();

// --------------- calculates the fes first derivative ---------------
   std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getFesGridPntr());
   for(Grid::index_t l=0; l<getFesGridPntr()->getSize(); l++) {
     std::vector<double> args = getFesGridPntr()->getPoint(l);
     unsigned int nargs = args.size();
     std::vector<double> args_values_trsfrm(nargs);
     std::vector< std::vector <double> > basisf(nargs);
     std::vector< std::vector <double> > derivs(nargs);
     bool all_inside = true;
     for(unsigned int k=0; k<nargs; k++) {
       basisf[k].assign(nbasisf[k],0.0);
       derivs[k].assign(nbasisf[k],0.0);
       pntrs_to_basisf_[k]->getAllValues(args[k],args_values_trsfrm[k],all_inside,basisf[k],derivs[k]);
     }
     std::vector<double> grad_fes_value;
     grad_fes_value.assign(nargs,0.0);
     std::vector<double> derivs_curr;
     for(size_t i=rank; i<ncoeffs; i+=stride){
       std::vector<unsigned int> indices=fes_coeffs_pntr_->getIndices(i);
       double coeff = fes_coeffs_pntr_->getValue(i);
       for(unsigned int k=0; k<nargs; k++) {
         derivs_curr.assign(nargs,1.0);
         for(unsigned int j=0; j<nargs; j++) {
	         if(j!=k){
	            derivs_curr[k]*=basisf[j][indices[j]];
	         }
	         else{
	            derivs_curr[k]*=derivs[j][indices[j]];
	         }
	       }
	       grad_fes_value[k]+=coeff*derivs_curr[k];
       }
     }
     if(stride!=1){
       for(unsigned int k=0; k<nargs; k++) {
         comm_in.Sum(grad_fes_value[k]);
       }
     }
     double value = 0.0;
     for(unsigned int k=0; k<nargs; k++) {
       value=getBeta()*grad_fes_value[k];
     }
     derv_grid_pntr_->setValue(l,value); //NOTE: derv_grid_pntr = beta * dF/ds
   }

// --------------- prints the derv_grid_pntr (if requested) ---------------
   std::string time;
   Tools::convert(getPntrToVesBias()->getIterationFilenameSuffix(),time);
   if(pstride!= 0 && getPntrToVesBias()->getIterationCounter()%pstride==0){
     std::string gradFes_fname = derv_init_filename_+time+".data";
     OFile ofile;
     ofile.link(*this);
     ofile.enforceBackup();
     ofile.open(gradFes_fname);
     derv_grid_pntr_->writeToFile(ofile);
     ofile.close();
   }
}

void TD_TS_TARGET::calculateSecondDerv(){
// --------------- initializes quantities ---------------
  std::vector<double> dx_ = getFesGridPntr()->getDx();
  std::vector<unsigned> nbins = pntr_to_bias_expansion_->getGridBins(); 
  std::vector<bool> isperiodic = getFesGridPntr()->getIsPeriodic();
  if(derv2_grid_pntr_==NULL){
    std::vector<std::string> min = getFesGridPntr()->getMin();
    std::vector<std::string> max = getFesGridPntr()->getMax();
    std::vector<std::string> args = getFesGridPntr()->getArgNames();
    derv2_grid_pntr_ = new Grid("gradient-order2",args,min,max,nbins,false,false,true,isperiodic,min,max);
    switch_grid_pntr_ = new Grid("switch",args,min,max,nbins,false,false,true,isperiodic,min,max);
  }
  nbins = getFesGridPntr()->getNbin();
// --------------- calculates the fes second derivative by numerically differentiating the fes first derivative ---------------
  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
    std::vector<unsigned int> ind = getFesGridPntr()->getIndices(l);
    std::vector<unsigned int> ind_c = getFesGridPntr()->getIndices(l);
    double grad = 0.0;
    double value;
    for(unsigned int k=0; k<getFesGridPntr()->getDimension(); k++) {
      double val1 = 0.0;
      double val2 = 0.0;
      if((ind[k]+nbins[k])%nbins[k]==0){                 //first point
        if(isperiodic[k]){
	  ind_c[k]=ind[k]+1;
	  val1 = derv_grid_pntr_->getValue(ind_c);
	  ind_c[k]=ind[k]+nbins[k]-1;
	  val2 = derv_grid_pntr_->getValue(ind_c);
	  value = (val1 - val2)/(2*dx_[k]);
	}
	else{
	  ind_c[k]= ind[k]+1;
	  val1 = derv_grid_pntr_->getValue(ind);
	  val2 = derv_grid_pntr_->getValue(ind_c);
	  value = (val2 - val1)/dx_[k];
	}
       }
       else if((ind[k]+1)%nbins[k]==0){                  //last point
         if(isperiodic[k]){ 
	   ind_c[k]=ind[k]-nbins[k]+1;
	   val1 = derv_grid_pntr_->getValue(ind_c);
	   ind_c[k]=ind[k]-1;
	   val2 = derv_grid_pntr_->getValue(ind_c);
	   value = (val1 - val2)/(2*dx_[k]);
	 }
	 else{
	   ind_c[k] = ind[k]-1;
	   val1 = derv_grid_pntr_->getValue(ind);
	   val2 = derv_grid_pntr_->getValue(ind_c);
	   value = (val1 - val2)/dx_[k];
	  }
        }
       else{
	 ind_c[k]=ind[k]+1;
	 val1 = derv_grid_pntr_->getValue(ind_c);
	 ind_c[k]=ind[k]-1;
	 val2 = derv_grid_pntr_->getValue(ind_c);
	 value = (val1 - val2)/(2*dx_[k]);
       }
       grad+= (value);
    }
    derv2_grid_pntr_->setValue(l,grad);
    double switchingfunction = 1+(exp((grad)*lambda));
    switchingfunction = 1/(switchingfunction);
    if (limit!=0.0){
      switchingfunction=((switchingfunction*(1-limit))+limit);
    }
    switch_grid_pntr_->setValue(l,switchingfunction);
  }
// --------------- prints the derv2_grid_pntr (if requested) ---------------
   std::string time;
   Tools::convert(getPntrToVesBias()->getIterationFilenameSuffix(),time);
   if(pstride!= 0 && getPntrToVesBias()->getIterationCounter()%pstride==0){
     std::string gradFes_fname = derv2_init_filename_+time+".data";
     OFile ofile;
     ofile.link(*this);
     ofile.enforceBackup();
     ofile.open(gradFes_fname);
     derv2_grid_pntr_->writeToFile(ofile);
     ofile.close();
   }
   
// --------------- prints the switch_grid_pntr (if requested) ---------------
   if(pstride!= 0 && getPntrToVesBias()->getIterationCounter()%pstride==0){
     std::string gradFes_fname = switch_init_filename_+time+".data";    
     OFile ofile;
     ofile.link(*this);
     ofile.enforceBackup();
     ofile.open(gradFes_fname);
     switch_grid_pntr_->writeToFile(ofile);
     ofile.close();
   }
}

}
}
