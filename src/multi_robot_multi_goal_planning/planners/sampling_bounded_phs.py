"""Uniform rejection sampling from an ellipsoid intersected with joint bounds.

Proposals use the smaller of the ellipsoid and its bounded box.
Rejected points are discarded, never clipped.
"""
import math
import numpy as np
from numba import njit


def joint_bounds(env):
    """A reversed interval denotes an unbounded coordinate; equal bounds fix it."""
    limits = np.asarray(env.limits, dtype=float)
    if np.any(np.isnan(limits)):
        raise ValueError("Joint limits must not contain NaN")
    bounded = limits[0] <= limits[1]
    return np.where(bounded, limits[0], -np.inf), np.where(bounded, limits[1], np.inf)


@njit(cache=True)
def _endpoint_geometry(a, b, cost_bound, lower, upper):
    """Fuse small-array geometry work; do not draw randomness here."""
    dim = len(a)
    delta = b-a
    distance = np.sqrt(np.sum(delta*delta))
    if (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or not np.isfinite(cost_bound) or cost_bound < 0):
        raise ValueError('Ellipsoid parameters must be finite and nonnegative')
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError('Invalid bounds')
    if cost_bound < distance:
        raise ValueError('Cost bound is smaller than focal distance')
    center = a/2+b/2
    singular = np.full(dim, np.sqrt((cost_bound-distance)/2)*np.sqrt((cost_bound+distance)/2))
    singular[0] = cost_bound/2
    u = np.eye(dim)
    if distance > 0:
        v = delta/distance
        v[0] += np.copysign(1., v[0])
        v /= np.sqrt(np.sum(v*v))
        for i in range(dim):
            for j in range(dim):
                u[i,j] -= 2*v[i]*v[j]
    extent = np.empty(dim)
    for i in range(dim):
        total = 0.
        for j in range(dim):
            total += (u[i,j]*singular[j])**2
        extent[i] = np.sqrt(total)
    lo,hi = np.maximum(lower,center-extent),np.minimum(upper,center+extent)
    disjoint = bool(np.any(lo>hi))
    rank = int(np.sum(singular>0))
    basis = np.empty((dim,rank))
    col = 0
    for j in range(dim):
        if singular[j]>0:
            for i in range(dim):
                basis[i,col] = u[i,j]*singular[j]
            col += 1
    use_box = False
    if not disjoint and rank==dim and rank>0 and np.all(hi>lo):
        log_ellipsoid = dim/2*math.log(math.pi)-math.lgamma(dim/2+1)+np.log(singular).sum()
        use_box = np.log(hi-lo).sum()<log_ellipsoid
    return center,singular,u,lo,hi,basis,rank,disjoint,use_box


@njit(cache=True)
def _ellipsoid_batch(unit, radii, basis, center, lower, upper):
    rank,batch = unit.shape
    dim = len(center)
    x = np.empty((dim,batch))
    valid = np.ones(batch,dtype=np.bool_)
    for k in range(batch):
        norm = 0.
        for j in range(rank):
            norm += unit[j,k]**2
        norm = np.sqrt(norm)
        if norm == 0:
            valid[k] = False
        for j in range(rank):
            unit[j,k] = (unit[j,k]/norm if norm>0 else 0.)*radii[k]**(1./rank)
        for i in range(dim):
            value = center[i]
            for j in range(rank):
                value += basis[i,j]*unit[j,k]
            x[i,k] = value
            if value<lower[i] or value>upper[i]:
                valid[k] = False
    return x,valid


@njit(cache=True)
def _box_mask(x,u,center,singular):
    dim,batch=x.shape
    valid=np.empty(batch,dtype=np.bool_)
    for k in range(batch):
        total=0.
        for j in range(dim):
            z=0.
            for i in range(dim):
                z+=u[i,j]*(x[i,k]-center[i])
            total+=(z/singular[j])**2
        valid[k]=total<=1.
    return valid


def _draw_counts(n, max_draws):
    if n < 0 or max_draws < 0 or int(n) != n or int(max_draws) != max_draws:
        raise ValueError('Draw counts must be nonnegative integers')
    return int(n), int(max_draws)


class PreparedBoundedPHS:
    """Immutable proposal geometry for one endpoint pair and one robot.

    Input arrays are copied. Keep instances local to the current informed
    interval: a new path, cost bound or joint bounds requires a new instance.
    The SVD and proposal-volume comparison are independent of sample draws.
    """

    def __init__(self, transform, center, lower, upper):
        transform = np.asarray(transform, dtype=float)
        center = np.asarray(center, dtype=float)
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        dim = len(center)
        if transform.shape != (dim, dim) or lower.shape != center.shape or upper.shape != center.shape:
            raise ValueError('Incompatible ellipsoid/bounds shapes')
        if not np.all(np.isfinite(transform)) or not np.all(np.isfinite(center)):
            raise ValueError('Ellipsoid parameters must be finite')
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError('Invalid bounds')
        self.center, self.lower, self.upper = center.copy(), lower.copy(), upper.copy()
        self.dim = dim
        extent = np.linalg.norm(transform, axis=1)
        self.lo = np.maximum(lower, center-extent)
        self.hi = np.minimum(upper, center+extent)
        self.disjoint = bool(np.any(self.lo > self.hi))
        self.use_box = False
        if not self.disjoint:
            self.u, self.singular, _ = np.linalg.svd(transform, full_matrices=False)
        self._finish_geometry()

    @classmethod
    def from_endpoints(cls, a, b, cost_bound, lower, upper):
        """Prepare a prolate hyperspheroid analytically, without either SVD.

        A Householder basis maps the first coordinate to the focal direction.
        Its sign/orientation does not affect the uniform ball distribution, but
        draws are not bitwise identical to the general SVD implementation.
        Equal foci, a zero minor radius and one-dimensional segments are valid.
        """
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (a.ndim != 1 or not len(a) or b.shape != a.shape
                or lower.shape != a.shape or upper.shape != a.shape):
            raise ValueError('Incompatible ellipsoid/bounds shapes')
        obj = cls.__new__(cls)
        obj.dim = len(a)
        (obj.center,obj.singular,obj.u,obj.lo,obj.hi,obj.basis,obj.rank,
         obj.disjoint,obj.use_box) = _endpoint_geometry(a,b,float(cost_bound),lower,upper)
        obj.lower, obj.upper = lower.copy(), upper.copy()
        for value in vars(obj).values():
            if isinstance(value,np.ndarray):
                value.setflags(write=False)
        return obj

    def _finish_geometry(self):
        dim = self.dim
        if not self.disjoint:
            # Preserve every positive axis, including arbitrarily thin ones.
            positive = self.singular > 0
            self.rank = int(positive.sum())
            self.basis = self.u[:, positive] * self.singular[positive]
            if self.rank == dim and self.rank > 0 and np.all(self.hi > self.lo):
                log_ellipsoid = (dim/2*math.log(math.pi) - math.lgamma(dim/2+1)
                                 + np.log(self.singular).sum())
                self.use_box = np.log(self.hi-self.lo).sum() < log_ellipsoid
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def sample(self, n=1, max_draws=4096):
        n, max_draws = _draw_counts(n, max_draws)
        stats = dict(proposal='ellipsoid', drawn=0, accepted=0)
        dim, center, lower, upper = self.dim, self.center, self.lower, self.upper
        empty = np.empty((dim, 0))
        if n == 0 or max_draws == 0 or self.disjoint:
            return empty, stats
        rank = self.rank
        if rank == 0:
            return np.repeat(center[:, None], n, axis=1), dict(stats, accepted=n)
        if self.use_box:
            stats['proposal'] = 'box'
        chunks = []
        count = 0
        while count < n and stats['drawn'] < max_draws:
            batch = min(max(32, 2*(n-count)), max_draws-stats['drawn'])
            if self.use_box:
                x = np.random.uniform(self.lo[:, None], self.hi[:, None], size=(dim, batch))
                valid = _box_mask(x,self.u,center,self.singular)
            else:
                unit = np.random.normal(size=(rank, batch))
                x,valid = _ellipsoid_batch(unit,np.random.random(batch),self.basis,center,lower,upper)
            stats['drawn'] += batch
            accepted = x[:, valid][:, :n-count]
            chunks.append(accepted)
            count += accepted.shape[1]
        stats['accepted'] = count
        return (chunks[0] if len(chunks)==1 else np.concatenate(chunks,axis=1)
                if chunks else empty), stats


def warm_bounded_phs_kernels():
    """Compile before planner timing without consuming either RNG stream."""
    a=np.zeros(2); b=np.ones(2)
    _endpoint_geometry(a,b,3.,-b,b)
    matrix=np.eye(2); center=a.copy(); radius=b.copy()
    lower=-b; upper=b.copy()
    for value in (matrix,center,radius,lower,upper):
        value.setflags(write=False)
    _ellipsoid_batch(np.ones((2,2)),b,matrix,center,lower,upper)
    _box_mask(np.ones((2,2)),matrix,center,radius)


warm_bounded_phs_kernels()


def sample_bounded_phs_with_given_matrices(transform, center, lower, upper,
                                         n=1, max_draws=4096):
    _draw_counts(n, max_draws)
    return PreparedBoundedPHS(transform, center, lower, upper).sample(n, max_draws)
