"""Capacity-preserving correction of integrated, closed-column face fluxes.

Only transfers change: a rejected parcel stays in its donor, including its
enthalpy. Stored water and energy are always rebuilt from the same faces.
The optional array backend keeps the arithmetic on a selected device.
"""


def bounded_interface_transfers(water,capacity,transfer,*,xp=None,roundoff_only=True):
    """Reduce directed faces until both exact and stored pore bounds hold.

    A line of directed faces has no cycles. Reducing an incoming face for a
    full cell, or an outgoing face for a dry cell, propagates the constraint
    toward its source without reversing a transfer. This is not a storage
    projection. The default rejects changes larger than the original
    Richards nonlinear/precision scale, so it cannot conceal a bad solve.
    """
    if xp is None:
        import numpy as xp
    water,capacity,transfer=[xp.asarray(v) for v in (water,capacity,transfer)]
    if (water.ndim!=1 or not len(water) or capacity.shape!=water.shape
            or transfer.shape!=(len(water)-1,) or water.dtype not in (xp.float32,xp.float64)
            or capacity.dtype!=water.dtype or transfer.dtype!=water.dtype):
        raise ValueError('Bounded faces require matching floating water/capacity(L), transfer(L-1)')
    if (not all(bool(xp.all(xp.isfinite(v))) for v in (water,capacity,transfer))
            or bool(xp.any((water<0.)|(water>capacity)))):
        raise ValueError('Bounded faces require valid original water and capacities')
    # A float32 water/flux sum is exact in this promoted coordinate. NumPy's
    # extended precision additionally resolves sub-ULP float64 endpoints.
    wide=xp.longdouble if xp.__name__=='numpy' else xp.float64
    old=water.astype(wide);maximum=capacity.astype(wide);face=transfer.copy()
    z=xp.zeros(1,water.dtype);zw=xp.zeros(1,wide)
    for _ in range(8*len(water)**2+1):
        f=face.astype(wide)
        exact=old+xp.concatenate((zw,f))-xp.concatenate((f,zw))
        stored=(water+xp.concatenate((z,face)))-xp.concatenate((face,z))
        bad=(exact<0.)|(exact>maximum)|(stored<0.)|(stored>capacity)
        if not bool(xp.any(bad)):
            if roundoff_only:
                scale=xp.maximum(xp.maximum(xp.max(xp.abs(water)),xp.max(capacity)),1.)
                bound=xp.maximum(1e-9,8.*xp.finfo(water.dtype).eps*scale)
                if bool(xp.any(xp.abs(face.astype(wide)-transfer.astype(wide))>bound)):
                    raise RuntimeError('Richards face correction exceeds nonlinear/roundoff scale')
            return face
        i=int(xp.argmax(bad));wet=bool((exact[i]>maximum[i])|(stored[i]>capacity[i]))
        excess=(xp.maximum(exact[i]-maximum[i],(stored[i]-capacity[i]).astype(wide))
                if wet else xp.maximum(-exact[i],-stored[i].astype(wide)))
        candidates=[]
        if i>0 and bool(face[i-1]!=0.) and bool(face[i-1]>0.)==wet:candidates.append(i-1)
        if i<len(face) and bool(face[i]!=0.) and bool(face[i]<0.)==wet:candidates.append(i)
        if not candidates:raise RuntimeError('No directed face can restore valid storage')
        j=max(candidates,key=lambda k:float(xp.abs(face[k])))
        magnitude=xp.abs(face[j]).astype(wide);target=xp.maximum(magnitude-excess,0.)
        value=(xp.sign(face[j]).astype(wide)*target).astype(water.dtype)
        if bool(xp.abs(value).astype(wide)>target) or bool(value==face[j]):
            value=xp.nextafter(value,xp.asarray(0.,water.dtype))
        if not bool(xp.abs(value)<xp.abs(face[j])):
            raise RuntimeError('Directed face correction made no progress')
        if xp.__name__.startswith('jax'):face=face.at[j].set(value)
        else:face[j]=value
    raise RuntimeError('Directed face correction did not terminate')


def close_integrated_transport(water,capacity,net,forward,reverse,*,heat=None,
                               forward_heat=None,reverse_heat=None,xp=None,
                               roundoff_only=True):
    """Return capacity-bounded mass faces and their consistent donor heat.

    Gross directional heat integrals retain different donor temperatures
    when a face reverses. An endpoint correction removes the proportional
    parcel from that direction, retaining it in its original donor cell.
    Valid faces and the original net heat remain byte-for-byte unchanged.
    """
    if xp is None:
        import numpy as xp
    fixed=bounded_interface_transfers(water,capacity,net,xp=xp,roundoff_only=roundoff_only)
    delta=net-fixed
    if not bool(xp.any(delta!=0.)):return fixed,forward,reverse,heat
    positive=xp.maximum(delta,0.);negative=xp.maximum(-delta,0.)
    scale=xp.maximum(xp.maximum(xp.abs(net),forward+reverse),1.)
    tolerance=8.*xp.finfo(net.dtype).eps*scale
    if bool(xp.any((positive>forward+tolerance)|(negative>reverse+tolerance))):
        raise RuntimeError('Net face correction has no matching directional parcel')
    if heat is not None:
        if forward_heat is None or reverse_heat is None:
            raise ValueError('Directional heat integrals are required for an enthalpy correction')
        hf=forward_heat/xp.where(forward>0.,forward,1.)
        hr=reverse_heat/xp.where(reverse>0.,reverse,1.)
        heat=(heat-positive*hf)+negative*hr
    return fixed,xp.maximum(forward-positive,0.),xp.maximum(reverse-negative,0.),heat
