import gurobipy as gpy
import math
import itertools as it
import numpy as np

from shadowlearn.kernel import GaussianKernel

def chop(x, minimum, maximum, tolerance=1e-4):
    '''Chops a number when it is sufficiently close to the extreme of
   an enclosing interval.

Arguments:

- x: number to be possibily chopped
- minimum: left extreme of the interval containing x
- maximum: right extreme of the interval containing x
- tolerance: maximum distance in order to chop x

Returns: x if it is farther than tolerance by both minimum and maximum;
         minimum if x is closer than tolerance to minimum
         maximum if x is closer than tolerance to maximum

Throws:

- ValueError if minimum > maximum or if x does not belong to [minimum, maximum]

'''
    if minimum > maximum:
        raise ValueError('Chop: interval extremes not sorted')
    if  x < minimum or x > maximum:
        raise ValueError('Chop: value not belonging to interval')

    if x - minimum < tolerance:
        x = 0
    if maximum - x < tolerance:
        x = maximum
    return x

def solve_optimization(x, mu, c=1.0, d=1.0, phi= 1.0, k=GaussianKernel(),
                       tolerance=1e-4, adjustment=0):
    '''Builds and solves the constrained optimization problem on the basis
   of the shadowed set learning procedure.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the slack trade-off in optimization
- d: constant managing the fuzziness grade trade-off in optimization
- phi: constant managing the adjustment for maximum distance between center and point image
- k: kernel function to be used
- tolerance: tolerance to be used in order to clamp the problem solution to
             interval extremes
- adjustment: diagonal adjustment in order to deal with non PSD matrices

Returns: a lists containing the optimal values for the two sets of independent
         variables chis of the problem

Throws:

- ValueError if either b, d or phi is non-positive or if x and mu have different lengths,
  or if d = (phi-1)/phi

'''
    if phi <= 0:
        raise ValueError('phi should be positive')
    if c <= 0:
        raise ValueError('c should be positive')
    if d < 0:
        raise ValueError('d should be non-negative')
    if phi != 1 and chop(d - phi/(phi-1), 0, 100) == 0:
        raise ValueError('d should be different from phi/(phi-1)')

    if len(x) != len(mu):
        raise ValueError('patterns and labels have different length')

    m = len(x)

    mu = np.array(mu)

    model = gpy.Model('possibility-learn')
    model.setParam('OutputFlag', 0)

    for i in range(m):
        if c < np.inf:
            model.addVar(name='eps_%d' % i, lb=-c*(1-mu[i]), ub=c*mu[i],
                         vtype=gpy.GRB.CONTINUOUS)

        else:
            model.addVar(name='eps_%d' % i, vtype=gpy.GRB.CONTINUOUS)
        
        model.addVar(name='beta_%d' % i, lb=0., vtype=gpy.GRB.CONTINUOUS)

    model.update()
    
    eps = [model.getVarByName('eps_%d' % i) for i in range(m)]
    betas = [model.getVarByName('beta_%d' % i) for i in range(m)]

    obj = gpy.QuadExpr()

    for i, j in it.product(range(m), range(m)):
        obj.add((eps[i] + betas[i]) * (eps[j] + betas[j]), k.compute(x[i], x[j]) / (1. - d*(1.-1./phi)))

    for i in range(m):
        obj.add(-1 * (eps[i] + betas[i]) * k.compute(x[i], x[i]))

    if adjustment:
        for i in range(m):
            obj.add(adjustment * (eps[i] + betas[i]) * (eps[i] + betas[i]))

    model.setObjective(obj, gpy.GRB.MINIMIZE)

    constEqual_eps = gpy.LinExpr()
    constEqual_eps.addTerms([1.0]*m, eps)
    #constEqual_eps.add(sum(eps), 1.0)
    model.addConstr(constEqual_eps, gpy.GRB.EQUAL, 1-d)
    
    constEqual_betas = gpy.LinExpr()
    #constEqual_betas.add(sum(betas), 1.0)
    constEqual_betas.addTerms([1.0]*m, betas)
    model.addConstr(constEqual_betas, gpy.GRB.EQUAL, d/float(phi))
    
    model.optimize()


    if model.Status != gpy.GRB.OPTIMAL:
        raise ValueError('optimal solution not found!')
        

    eps_opt = np.array([chop(e.x, l, u, tolerance)
                for e, l, u in zip(eps, -c*(1-np.array(mu)), c*np.array(mu))])
    
    betas_opt = np.array([chop(b.x, 0, np.inf, tolerance) for b in betas])

    return (eps_opt, betas_opt)

    
def shadowed_learn(x, mu, c=1.0, d=1.0, phi= 1.0, k=GaussianKernel(), adjustment=0):
    '''Induces a fuzzy set membership function.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the slack trade-off in optimization
- d: constant managing the fuzziness grade trade-off in optimization
- phi: constant managing the adjustment for maximum distance between center and point image
- k: kernel function to be used
- adjustment: diagonal adjustment in order to deal with non PSD matrices

Returns: (f, e) with f being a function associating to a generic object the
         inferred degree of membership, and e being an estimate of the error

'''

    #x = map(np.array, x)

    try:
        eps, betas = solve_optimization(x, mu, c=c, d=d, phi=phi, k=k,
                                    tolerance=1e-4, adjustment=adjustment)
    except ValueError:
        return (None, np.inf)

    gram = np.array([[k.compute(x1, x2) for x1 in x] for x2 in x])
    fixed_term = (1-d*(1.-1./phi))**-2 * np.array(eps+betas).dot(gram.dot(eps+betas))

    def estimated_square_distance_from_center(x_new):
        return k.compute(x_new, x_new) \
               - 2.0 / (1-d*(1.-1./phi)) * np.array([k.compute(x_i, x_new) for x_i in x]).dot(eps+betas) \
               + fixed_term
    
    eps_SV_index = [i for i in range(len(eps))
                    if (-c*(1-mu[i]) < eps[i] < 0) or (0 < eps[i] < c*mu[i])]
    
    betas_SV_index = [i for i in range(len(betas)) if betas[i] > 0]
    
    betas_SV_square_distance = [estimated_square_distance_from_center(x[i])
                                for i in betas_SV_index]
    
    eps_SV_square_distance = [estimated_square_distance_from_center(x[i])
                              for i in eps_SV_index]

    if len(eps_SV_square_distance) == 0:
        return (None, np.inf)

    R2 = np.mean(eps_SV_square_distance)

    if len(betas_SV_square_distance) == 0:
        return (None, np.inf)

    M = np.mean(betas_SV_square_distance)
    
    def fuzzy_membership(x):
        r = estimated_square_distance_from_center(np.array(x))
        result = 1. - (r - R2) / (M - R2)
        
        if r <= R2:
            return 1
        if r >= M:
            return 0
        if result < 0:
            print 'result:', result
            print 'x was', x
            print 'distance was', r
            print 'R2 was', R2
            print 'M was', M
        return result


        # return 1 if r <= R2 \
        #         else (0 if r >= M else result)

    train_err = np.mean([(fuzzy_membership(x_i) - mu_i)**2
                         for x_i, mu_i in zip(x, mu)])
    sqrt = 1-math.sqrt(0.5)
    x_up = R2 + sqrt * (M - R2)
    x_dw = R2 + (1-sqrt) * (M - R2)
    
    def shadowed_membership(x):
        return 1 if estimated_square_distance_from_center(x) <= x_up \
                 else (0 if estimated_square_distance_from_center(x) > x_dw else 0.5)

    return (fuzzy_membership, train_err, shadowed_membership)
