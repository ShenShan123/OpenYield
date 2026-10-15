"""Route component ports while preserving physical obstacles and clearance."""
from .router import RoutingError


def route_compact(router, routed, order, attempts=8):
    """Negotiate pin access by moving a caged net ahead of its blockers.

    Every retry restores all physical obstacles and terminal reservations.
    Only route order changes; no clearance or circuit constraint is relaxed.
    """
    owner, blocked = dict(router.owner), set(router.blocked)
    halos = {p:set(nets) for p,nets in router.via_halos.items()}
    history = []
    seen = set()
    for attempt in range(attempts):
        identity = tuple(order)
        if identity in seen:
            raise RuntimeError('Compact routing repeated a failed order; expand placement spacing')
        seen.add(identity)
        router.owner, router.blocked = dict(owner), set(blocked)
        router.paths, router.vias = [], []
        router.via_halos = {p:set(nets) for p,nets in halos.items()}
        try:
            for index,net in enumerate(order):
                points=list(dict.fromkeys(router.point(*p) for p in routed[net]))
                if len(points)>1:
                    router.connect(net,points[0],points[1:])
                if index%50==0:
                    print('routed',index,'/',len(order),net,'attempt',attempt+1,flush=True)
            return history
        except RoutingError as error:
            candidates = [order.index(n) for n in error.blockers if n in order[:index]]
            if not candidates or attempt+1 == attempts:
                raise
            position = min(candidates)
            history.append(dict(net=error.net, terminal=error.target,
                                blockers=sorted(error.blockers), attempt=attempt+1))
            order.remove(error.net)
            order.insert(position,error.net)
            print('retry routing:',error.net,'before',order[position+1],flush=True)


def compact_route_order(routed):
    """Connect small nets before global trunks can enclose local terminals."""
    def key(net):
        points=set(routed[net])
        count=len(points)
        local=count<=12
        span=sum(max(p[axis] for p in points)-min(p[axis] for p in points) for axis in (0,1))
        return not local, net not in ('VDD','VSS'), span if local else -count, count, net
    return sorted(routed,key=key)
