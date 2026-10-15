"""Fail closed when an immutable physical plan differs from its source MOS."""
def require_devices(component, expected):
    if component.children or {d.name for d in component.devices}!=set(expected):
        raise ValueError(f'{component.name}: physical device inventory mismatch')
    for d in component.devices:
        nodes,kind,width=expected[d.name]
        if (d.nodes!=tuple(nodes.split()) or d.model!=kind+'MOS_VTG' or d.fingers!=1 or
                abs(d.width_m*1e9-width)>1e-6 or abs(d.length_m*1e9-50)>1e-6):
            raise ValueError(f'{component.name}/{d.name}: physical plan requires the frozen topology and W/L')


def core_contract(component,kind):
    e={'MPGL':('BL WL Q VSS','N',135),'MPGR':('BLB WL QB VSS','N',135),
       'MPDL':('Q QB VSS VSS','N',205),'MPDR':('QB Q VSS VSS','N',205),
       'MPUL':('Q QB VDD VDD','P',90),'MPUR':('QB Q VDD VDD','P',90)}
    if kind!='cell':
        e={n:(nodes.replace('QB','VDD').replace('BLB','RBLB').replace('BL ','RBL '),k,w) for n,(nodes,k,w) in e.items()}
    if kind=='passive':
        e={n:(' '.join([nodes.split()[0],'VSS',*nodes.split()[2:]]) if n not in ('MPGL','MPGR') else nodes,k,w) for n,(nodes,k,w) in e.items()}
    if kind=='dummy':
        e['MPGL']=('bl_node_noconnect WL Q VSS','N',135)
        e['MPGR']=('blb_node_noconnect WL VDD VSS','N',135)
    require_devices(component,e)

PRECHARGE={'M1':('BL ENB VDD VDD','P',270),'M2':('BLB ENB VDD VDD','P',270),'M3':('BL ENB BLB VDD','P',270)}
SENSE={'M1':('Q QB net1 VSS','N',270),'M2':('Q QB VDD VDD','P',540),
       'M3':('QB Q net1 VSS','N',270),'M4':('QB Q VDD VDD','P',540),
       'M5':('Q ISO IN VDD','P',720),'M6':('QB ISO INB VDD','P',720),'M7':('net1 EN VSS VSS','N',270)}
WRITE={'M1':('DINB DIN VDD VDD','P',360),'M2':('DINB DIN VSS VSS','N',180),
       'M3':('ENB EN VDD VDD','P',360),'M4':('ENB EN VSS VSS','N',180),
       'M5':('int1 DINB VDD VDD','P',720),'M6':('BL ENB int1 VDD','P',720),
       'M7':('BL EN int2 VSS','N',360),'M8':('int2 DINB VSS VSS','N',360),
       'M9':('int3 DIN VDD VDD','P',720),'M10':('BLB ENB int3 VDD','P',720),
       'M11':('BLB EN int4 VSS','N',360),'M12':('int4 DIN VSS VSS','N',360)}
