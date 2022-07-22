# -*- coding: utf-8 -*-
"""
Contain the method used to generate all combinations of hyperparameters for a grid search.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import itertools


def unfold_params(params = {}):
    """ Given a dictionary of hyperparameters, this method returns a list where
        each element is a dictionary of pairs <hyperparameter name, array hyperparameter value>,
        and each dictionary represents a possible combination of hyperparameter.

    Parameters
    ----------
    params (dict): The hyperparameter dictionary. The dictionary must be of the format
                   {
                       'fixed': [
                           {
                               list of couples <hyperparameter name, array of hyperparameter value>
                           }
                           {
                               list of couples <hyperparameter name, array of hyperparameter value>
                           }
                           (...)
                           {
                               list of couples <hyperparameter name, array of hyperparameter value>
                           }
                       ],
                       'variable': [
                           {
                               list of couples <hyperparameter name, array of hyperparameter value>
                           }
                           {
                               list of couples <hyperparameter name, array of hyperparameter value>
                           }
                           (...)
                           {
                               list of couples <hyperparameter name, array of hyperparameter value>
                           }
                       ]
                   }
                   
                   what this method essentially does is to create every possible
                   combination of hyperparameters for each dictionary separately.
                   Each combination is represented by a list where each element is
                   a tuple of the form <hyperparameter name, single hyperparameter value>.
                   Then, it stacks the various lists of the 'fixed' list, and performs
                   a cartesian product with the stacked lists in the 'variable' list
                   obtained in the same way of the "fixed" list.
                   
                   This approach allows us to test different architectures without having
                   to rewrite, for each architecture, the entire hyperparameters list,
                   but just the hyperparameters responsible for different architectures.
                   For example: if we want to test two architectures, one with 2 
                   layers and one with one layer, what we can do is:
                       
                   {
                       'fixed': [
                           {
                               'nn__max_epochs' : [100, 200]
                           }
                       ],
                       'variable': [
                           {
                               'nn__layers__0__n_units' : [10, 20]
                           }
                           {
                               'nn__layers__0__n_units' : [5, 2],
                               'nn__layers__1__n_units' : [4, 3]
                           }
                       ]
                   }
                   
                   As we can see, we don't need to rewrite the hyperparameter
                   max_epochs for each architecture. This becomes useful if we 
                   had tens of hyperparameters alongside it.

    Returns
    list: a list where each element is a dictionary of pairs 
        <hyperparameter name, hyperparameter value>, and each dictionary
        represents a possible hyperparameter combination obtained trough the 
        "params" parameter
    """
    assert(isinstance(params, dict)), "unfold_params: params must be a dictionary"
    fixed = []
    variable = []
    for keys, values in params.items():
        if(keys == 'fixed') :
            for x in values:
                k, val = zip(*x.items())
                combinations = [dict(zip(k, v)) for v in itertools.product(*val)]
                fixed.extend(combinations)
        elif(keys == 'variable'):
            for x in values:
                k, val = zip(*x.items())
                combinations = [dict(zip(k, v)) for v in itertools.product(*val)]
                variable.extend(combinations)
        else:
            print("error: unknown key in params dictionary")
            return {}
    
    if not fixed:
        return variable
    elif not variable:
        return fixed
    else:
        return ([{**l[0], **l[1]} for l in itertools.product(fixed, variable)])
