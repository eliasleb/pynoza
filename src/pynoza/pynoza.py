#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:16:58 2021

@author: Elias Le Boudec, elias.leboudec@epfl.ch
"""
import numpy as np
import sys, sympy

def set_default(dictionary,key,default_val):
    if key not in dictionary:
        dictionary[key] = default_val
    return dictionary

class Solution:
    """A class to compute solutions of Maxwell's equations, based on"""
    """time-domain multipole moments."""
    def __init__(self,
                max_order = 0,
                wave_speed = 1):
        """Initialize the solution class.
        
        Keyword arguments:
        max_order -- the maximum order to which multipole moments will be computed (default 0)
        wave_speed -- Wave speed `c` used to compute the retarded time t-r/c, in natural units (default 1)
        """"
        self.max_order = max_order
        self.c = wave_speed
        self._shape = (max_order+1,max_order+1,max_order+1)
        self._aux_func = dict()
        for ind, _ in np.ndenumerate(np.zeros(self._shape)):
            self._aux_func[tuple(ind)] = dict()
        self._aux_func[(0,0,0)] = {(0,0,0,0,1):1.}
        self.mu = 4*np.pi*1e-7
        
        self.ran_recurse = False
        self.ran_set_moments = False
        
    def _increase_order(self,known_index,index):
        """Private method to compute the auxiliary function.
        
        Positional arguments:
        known_index -- multi-index at which the auxiliary function has already been computed
        index -- multi-index to compute the auxiliary function"""
        known_dim = np.where(np.array(known_index)-np.array(index))[0][0]
        for signature in self._aux_func[known_index]:
            coeff = self._aux_func[known_index][signature]
            # We need to apply the recursion formula to this term.
            # This adds three new terms; two for the space-derivative,
            # and one for the time-derivative
            exponent_x_i = signature[known_dim]
            if exponent_x_i > 0:
                identity_first_term = list(signature)
                identity_first_term[known_dim] -= 1 # differentiate
                try:
                    self._aux_func[index][tuple(identity_first_term)] \
                        += coeff * exponent_x_i
                except KeyError:
                    self._aux_func[index][tuple(identity_first_term)] \
                        = coeff * exponent_x_i
            exponent_r = signature[-1]
            identity_secnd_term = list(signature)
            identity_secnd_term[known_dim] += 1 # numerator
            identity_secnd_term[-1] += 2 # denominator
            try:
                self._aux_func[index][tuple(identity_secnd_term)] -= \
                    coeff * exponent_r
            except KeyError:
                self._aux_func[index][tuple(identity_secnd_term)] = \
                    -coeff * exponent_r
            # Time-derivative term
            identity_third_term = list(signature)
            identity_third_term[known_dim] += 1
            identity_third_term[3] += 1 # time-derivative
            identity_third_term[-1]+= 1 # denominator
            try:
                self._aux_func[index][tuple(identity_third_term)] -= \
                coeff / self.c
            except KeyError:
                self._aux_func[index][tuple(identity_third_term)] = \
                -coeff / self.c

    def recurse(self, verbose=True):
        """Compute the auxiliary function up to the max order.
        
        Keyword arguments:
        verbose -- whether to print the computed multi-index (default True)"""
        self.ran_recurse = True
        for order in range(1,self.max_order+1):
            for ind, _ in np.ndenumerate(np.zeros((self._shape))):
                ind = np.array(ind)
                if np.sum(ind) == order:
                    known_ind = ind.copy()
                    known_ind[np.where(ind>0)[0][0]] -= 1
                    if verbose:
                        sys.stdout.write("\rComputing order {}...".format(order))
                    self._increase_order(tuple(known_ind),tuple(ind))
        if verbose:
            print("Done.")
        
    def _evaluate(self,ind,T,X1,X2,X3,R,Hs,**kwargs):
        """Evaluate the auxiliary function.
        
        Positional arguments:
        ind -- multi-index of the auxiliary function
        T -- evaluated time
        X1 -- evaluated first coordinate (aka x)
        X2 -- evaluated second coordinate (aka y)
        X3 -- evaluated third coordinate (aka z)
        R -- equal to X1**2+X2**2+X3**2. Passed to avoid computing it repeatedly.
        Hs -- dictionary of the derivatives of thetime-dependent excitation function.
              Must be in the form {order:derivative of order} for order=-1..max_order+2
              
        Keyword arguments:
        delayed -- whether to evaluate the auxiliary function in t or t-r/c (default True)"""
        Y = np.zeros(X1.shape)
        for signature in self._aux_func[ind]:
            if kwargs["delayed"]:
                dY = Hs[signature[3]](T-R/self.c)*self._aux_func[ind][signature]
            else:
                dY = Hs[signature[3]](T)*self._aux_func[ind][signature]
            if signature[0] > 0:
                dY *= X1**signature[0]
            if signature[1] > 0:
                dY *= X2**signature[1]
            if signature[2] > 0:
                dY *= X3**signature[2]
            if signature[-1] > 0:
                dY /= R**signature[-1]
            Y += dY
        return Y
    
    def set_moments(self,
                    current_moment=lambda a1,a2,a3:[0,0,0],
                    charge_moment=lambda a1,a2,a3:[0,0,0]):
        """Set the current and charge moment functions.
        
        Keyword arguments:
        current_moment -- a callable returning the current moment for a given multi-index a1,a2,a3
        charge_moment -- a callable returning the charge moment for a given multi-index a1,a2,a3"""
        self.ran_set_moments = True
        self.current_moment = current_moment
        self.charge_moment = charge_moment
        
    def compute_E_field(self,
                        x1,x2,x3,t,
                        h_sym,
                        t_sym,
                        **kwargs):
        """Compute the electric field from the moments.
        
        Positional arguments:
        x1,x2,x3 -- array of the spatial coordinates to evaluate the E-field at (aka x,y,z)
        t -- array of the time coordinates to evaluate the E-field at
        h_sym -- symbolic time-dependent function describing the shape of the current
        t_sym -- symbolic variable representing time, used in h_sym
        
        Keyword arguments:
        verbose -- whether to display the progress (default False)
        delayed -- whether to evaluate the field at the retarded time t-r/c (default True)"""
    
        if not self.ran_recurse or not self.ran_set_moments:
            raise RuntimeError("You must first run the `recurse' and `set_moments' methods.")
            
        np.seterr(divide="raise",
                 over="raise",
                 under="warn",
                 invalid="raise")
        kwargs = set_default(kwargs,"verbose",False)
        kwargs = set_default(kwargs,"delayed",True)

        X1,X2,X3,T = np.meshgrid(x1,x2,x3,t)
        R = np.sqrt(X1**2+X2**2+X3**2)
        
        thresh = 1e-14

        h_0_sym = h_sym.integrate(t_sym)
        Hs_sym = {-1:h_0_sym,0:h_sym}
        for order in range(1,self.max_order+3):
            if kwargs["verbose"]:
                sys.stdout.write("\rComputing derivative of order {}...".format(order))
            Hs_sym[order] = sympy.diff(Hs_sym[order-1]).simplify()
        if kwargs["verbose"]:
            print("Done.")
        Hs = dict()
        for order in Hs_sym:
            Hs[order] = sympy.lambdify(t_sym,Hs_sym[order])

        E = 0

        Hs_integral   = list()
        Hs_derivative = list()
        for order in range(-1,self.max_order+3):
            if order > 0:
                if order <= self.max_order:
                    Hs_derivative.append(Hs[order])
                    Hs_integral.append(Hs[order])
                else:
                    Hs_derivative.append(lambda t: 0)
                    Hs_integral.append(lambda t: 0)
            else:
                Hs_integral.append(Hs[order])


        for ind, _ in np.ndenumerate(np.zeros((self._shape))):

            ind = np.array(ind)
            if np.sum(ind) <= self.max_order:
                a1,a2,a3 = ind
                if kwargs["verbose"]:
                    sys.stdout.write("\rComputing index {}...".format(ind))
                charge_moment = -self.mu*self.c**2*np.array(
                    self.charge_moment(a1,a2,a3)).reshape((3,1,1,1,1))
                current_moment = -self.mu*np.array(
                    self.current_moment(a1,a2,a3)).reshape((3,1,1,1,1))
                if np.any(charge_moment) > thresh:
                    E += (-1)**np.sum(ind)/fact(ind)\
                            *charge_moment*self._evaluate(tuple(ind),
                                                 T,X1,X2,X3,R,
                                                 Hs_integral,
                                                 **kwargs)/4/np.pi      
                if np.any(current_moment) > thresh:
                    E += (-1)**np.sum(ind)/fact(ind)\
                            *current_moment*self._evaluate(tuple(ind),
                                                       T,X1,X2,X3,R,
                                                       Hs_derivative,
                                                       **kwargs)/4/np.pi
        if kwargs["verbose"]:
            print("Done.")
        return E
    
def fact(tuple_):
    res = 1
    for i in tuple_:
        res *= np.math.factorial(i)
    return res
