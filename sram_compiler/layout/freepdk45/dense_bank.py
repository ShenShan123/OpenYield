"""OpenRAM-pitch bank assembly; controller/input registers remain external."""
from collections import defaultdict
from pathlib import Path
from .bank import source_contract
from .components import Component,Instance,CellView,read_component
from .cell_flow import leaves
from .dense_core import make_core
from .dense_decoder import make_decoder
from .dense_columns import make_precharge,make_sense
from .dense_write import make_write
from .dense_aux import make_aux
from .dense_views import write_view
from .dense_slots import fold_class,make_write_slot,make_precharge_slot
from .lib import Builder,new_layout,pya,u


def split_row(rows):
    """Lower-half row count of a split bank, or None when it cannot split.

    Each half holds whole final slices, and for three decoder levels whole
    64-row predecode groups, so only level-0 enables cross the periphery.
    """
    group=64 if rows>64 else 8
    split=group*max(1,(rows//group+1)//2)
    return split if split<rows else None


def build(deck,output,include_auxiliary=True,folded=None,split=None,reserve=1000.,bridge=True,spare=0.):
    """split=row places rows [0, split) below the column periphery.

    The lower array's top edge lies ``reserve`` nm below the write band, which
    leaves room for the controller's data registers. Bitlines continue on M5
    through the periphery. With ``bridge`` the decoder nets shared by both
    halves cross the band as straight M5 tracks; otherwise both halves keep
    band-edge pins for the controller to join. spare widens the free decoder
    space of both halves (centre controller lanes).
    """
    source,core_name,specs,data_component,_=source_contract(deck,True)
    rows,cols=source.rows,source.cols
    if rows<8 or rows%8 or rows>512 or cols<8 or cols%2 or cols>512:
        raise ValueError('OpenRAM-pitch bank supports 8-512 rows (multiples of 8) and 8-512 even columns')
    if spare and split is None:raise ValueError('Spare decoder width applies to split banks')
    if split is not None and (include_auxiliary or folded is False or split!=split_row(rows) or reserve<1000):
        raise ValueError('A split bank uses the folded periphery, external auxiliaries and split_row(rows)')
    if split is not None:folded=True
    # Rows <= 32 keep the verified reference periphery; larger row classes
    # fold their enlarged precharge/write/buffer devices into slot rows.
    if folded is None:
        from .dense_contract import PRECHARGE, WRITE
        def widths_match(component: Component, reference: dict) -> bool:
            return not component.children and all(
                d.name in reference and abs(d.width_m*1e9-reference[d.name][2])<1e-6
                for d in component.devices)
        folded=(fold_class(rows) or not widths_match(specs['XPRECHARGE_0'],PRECHARGE)
                or not widths_match(specs['XWRITEDRIVER_0'],WRITE))
    ceiling=rows<=32 and cols<=32
    row_names=['XDECODER']+[f'XWL_DRV_{r}' for r in range(rows)]
    row_specs={n:read_component(deck,n) for n in row_names}
    specs.update(row_specs)
    aux_names=['XRWL','XD_LATCH'] if include_auxiliary else []
    specs.update({n:read_component(deck,n) for n in aux_names})
    omitted=[n for n in source.instances if '/' not in n and n not in specs]
    expected={'XTIME_CONTROL'} if include_auxiliary else {'XTIME_CONTROL','XRWL','XD_LATCH'}
    if set(omitted)!=expected:
        raise ValueError('Unexpected omitted source instances')
    ly=new_layout();b=Builder(ly,f'DENSE_BANK_{rows}X{cols}')
    if split is None:
        decoder=make_decoder(ly,source,row_specs);core=make_core(ly,source,specs,core_name)
    else:
        decoder=make_decoder(ly,source,row_specs,(split,rows),spare);core=make_core(ly,source,specs,core_name,(split,rows))
        decoder_lo=make_decoder(ly,source,row_specs,(0,split),spare);core_lo=make_core(ly,source,specs,core_name,(0,split))
    sa=make_sense(ly,specs['XSENSEAMP_0'])
    if folded:
        wslot=make_write_slot(ly,'DENSE_WRITE_SLOT',{'L':specs['XWRITEDRIVER_0'],'U':specs['XWRITEDRIVER_1']},
                              {'L':specs['XDIN_BUF_0'],'U':specs['XDIN_BUF_1']})
        wrep=make_write_slot(ly,'DENSE_WRITE_SLOT_RBL',{'U':specs['XREPLICA_WDRV_LOAD']},
                             {'U':specs['XDIN_BUF_0']},replica=True)
        pslot=make_precharge_slot(ly,'DENSE_PRECHARGE_SLOT',{'L':specs['XPRECHARGE_0'],'U':specs['XPRECHARGE_1']})
        prep=make_precharge_slot(ly,'DENSE_PRECHARGE_SLOT_RBL',{'U':specs['XPRECHARGE_RBL']},replica=True)
        # Band gaps keep wells, implants and M2 of the sense asset clear.
        # A p-well top boundary needs 225 nm to the sense asset's n-well.
        gap=300 if max(wslot.supplies,key=lambda t:t[1])[0]=='VDD' else 400
        sa_y=wslot.height+gap;pc_y=sa_y+sa.height_nm+140;cy=pc_y+pslot.height+1000
    else:
        pc=make_precharge(ly,specs['XPRECHARGE_0'])
        wd=make_write(ly,specs['XWRITEDRIVER_0']);buf=make_aux(ly,specs['XDIN_BUF_0'])
        cy=13920.
    rwl=make_aux(ly,specs['XRWL'],ceiling) if include_auxiliary else None
    latch=make_aux(ly,specs['XD_LATCH']) if include_auxiliary else None
    cx=decoder.width_nm+400
    terminals={};fingers=[];placed=[];pins={};points={}
    def wire(layer,x,y,xx,yy,w=None):
        w=w or (70 if int(layer[1:])<4 else 140 if layer=='m4' else 145)
        b.rect(layer,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    def via(n,x,y):
        if n==1:
            b.sq('v1',x,y,65);b.rect('m1',x-67.5,y-32.5,x+67.5,y+32.5);b.rect('m2',x-35,y-70,x+35,y+70)
        else:
            b.via(n,x,y)
            for k in (n,n+1):
                if k<4:b.rect(f'm{k}',x-35,y-70,x+35,y+70)
                else:b.sq(f'm{k}',x,y,140 if k==4 else 145)
    def lift(point,upper=4):
        x,y,l=point
        for n in range(min(int(l[1:]),upper),max(int(l[1:]),upper)):via(n,x,y)
        return x,y
    def add(v,x,y,prefix='',mapping=None,rotate=False):
        if rotate:b.cell.insert(pya.CellInstArray(v.builder.cell.cell_index(),pya.Trans(pya.Trans.R90,u(x),u(y))))
        else:b.place(v.builder,x,y)
        def tr(p):
            px,py,l=p;return (x-py,y+px,l) if rotate else (x+px,y+py,l)
        terminals.update({('/'.join(filter(None,(prefix,n)))):tr(p) for n,p in v.device_pins.items()})
        fs=getattr(v,'physical_fingers',None)
        if fs is None:
            fs=[dict(device='/'.join(filter(None,(p,d.name))),finger=i,width_nm=d.width_m*1e9/d.fingers,
                     length_nm=d.length_m*1e9,model=d.model,nodes=d.nodes)
                for p,c,_ in leaves(v.component) for d in c.devices for i in range(d.fingers)]
        for f in fs:
            f=dict(f,device='/'.join(filter(None,(prefix,f['device']))))
            if 'gate_xy_nm' in f:f['gate_xy_nm']=tr((*f['gate_xy_nm'],'poly'))[:2]
            fingers.append(f)
        pp={p:tr(pt) for p,pt in v.pins.items()}
        for p,pt in pp.items():points.setdefault(mapping[p] if mapping else p,[]).append(pt)
        placed.append(dict(instance=prefix or v.component.name,x_nm=x,y_nm=y,rotate_90=rotate,
                           width_nm=v.height_nm if rotate else v.width_nm,height_nm=v.width_nm if rotate else v.height_nm))
        return pp
    cp=add(core,cx,cy);dp=add(decoder,0,cy)
    if split is not None:
        # The lower array's top edge (yb) faces the data-register reserve.
        yb=-reserve;ylo=yb-split*1365
        cp_lo=add(core_lo,cx,ylo);dp_lo=add(decoder_lo,0,ylo)
        cp={**cp_lo,**cp};wl_pins={**dp_lo,**dp}
    else:wl_pins=dp
    for r in range(rows):wire('m3',*wl_pins[f'WL{r}'][:2],*cp[f'WL{r}'][:2])
    def vertical_m1(point):
        x,y,l=point;b.sq('v1',x,y,65)
        b.rect('m1',x-32.5,y-67.5,x+32.5,y+67.5);b.rect('m2',x-35,y-70,x+35,y+70)
        return x,y,'m2'
    right_root=cx+core.width_nm+400
    def root_x(net):
        if include_auxiliary:
            return {'VDD':50.,'VSS':-650.,'RWL':-300.}[net] if net in ('VDD','VSS','RWL') else 800.+300*control_order.index(net)
        return {'VDD':right_root,'VSS':right_root+300,'RWL':-300.}[net] if net in ('VDD','VSS','RWL') else cx-150
    def global_bar(net,point):
        x,y=lift(point);tx=root_x(net)
        if include_auxiliary and cols==8 and net==control_order[6]:tx=-1000.
        wire('m4',tx,y,x,y)
        global_sites.setdefault(net,[]).append((tx,y))
    control_order=['PRE','s_en','sa_iso','w_en','wl_en','RWL','SA_Q'+str(cols-1)]
    # Resolve names from the actual source boundary.
    control_order[:4]=[source.instances[n]['pins'][p] for n,p in
        [('XPRECHARGE_0','ENB'),('XSENSEAMP_0','EN'),('XSENSEAMP_0','ISO'),('XWRITEDRIVER_0','EN')]]
    global_sites={}
    # Decoder/core power and enable cross over to isolated left-side M6 trunks.
    if include_auxiliary:
        global_bar('VDD',dp['VDD']);global_bar('VSS',(200,cy+600,'m2'))
    else:
        px,py,pl=dp['VDD'];wire('m2',px,py,px,cy-600);global_bar('VDD',(px,cy-600,'m2'))
        wire('m2',200,cy+100,200,cy-300);global_bar('VSS',(200,cy-300,'m2'))
        if split is not None:
            # Mirror the decoder supply bars into the gap above the lower array.
            px,py,pl=dp_lo['VDD'];wire('m2',px,py,px,yb+600);global_bar('VDD',(px,yb+600,'m2'))
            px,py,pl=dp_lo['VSS'];wire('m2',px,py,px,yb+300);global_bar('VSS',(px,yb+300,'m2'))
    if include_auxiliary:
        wy=cy-650 if folded else cy-1000
        ex,ey,el=dp['wl_en'];wire('m5',ex,ey,ex,wy);global_bar('wl_en',(ex,wy,'m5'))
    else:pins['wl_en']=dp['wl_en']
    vx=cx+352.5
    ground_x=cx-(200 if include_auxiliary else 150)
    # Core VDD trunks join on M6 one row inside each array; VSS rails at the
    # periphery-facing edge drop to an M4 bar in the adjacent gap.
    for vy,edge,bar in [(cy+1700,cy,cy-300)]+([(yb-1700,yb,yb+300)] if split is not None else []):
        via(4,vx,vy);via(5,vx,vy);wire('m6',root_x('VDD'),vy,vx,vy)
        global_sites.setdefault('VDD',[]).append((root_x('VDD'),vy))
        wire('m1',ground_x,edge,cx+352.5,edge);wire('m1',ground_x,edge,ground_x,bar)
        global_bar('VSS',(ground_x,bar,'m1'))
    rx,ry=lift(cp['RWL'],6);wire('m6',-300,ry,rx,ry)
    global_sites.setdefault('RWL',[]).append((-300,ry))
    if folded:
        sas=[]
        def place_slot(view,x,y,roles):
            b.place(view.builder,x,y)
            for key,(px,py,l) in view.terminals.items():
                pre,rest=key.split('.',1);terminals[roles[pre]+'/'+rest]=(x+px,y+py,l)
            for f in view.fingers:
                pre,rest=f['device'].split('.',1);gx,gy=f['gate_xy_nm']
                fingers.append(dict(f,device=roles[pre]+'/'+rest,gate_xy_nm=(x+gx,y+gy)))
            for inst in roles.values():
                placed.append(dict(instance=inst,cell=view.builder.name,x_nm=x,y_nm=y,rotate_90=False,
                                   width_nm=view.width,height_nm=view.height))
            return {k:[(x+px,y+py,l) for px,py,l in v] for k,v in view.pins.items()}
        slot_pins={};supply=defaultdict(dict)
        for pair in range(cols//2+1):
            c=2*pair;x=cx+pair*1410;last=pair==cols//2
            if last:
                wroles={'U':'XREPLICA_WDRV_LOAD'};proles={'U':'XPRECHARGE_RBL'};wv,pv=wrep,prep
            else:
                wroles={'L':f'XWRITEDRIVER_{c}','U':f'XWRITEDRIVER_{c+1}','LB':f'XDIN_BUF_{c}','UB':f'XDIN_BUF_{c+1}'}
                proles={'L':f'XPRECHARGE_{c}','U':f'XPRECHARGE_{c+1}'};wv,pv=wslot,pslot
            wpins=place_slot(wv,x,0,wroles);ppins=place_slot(pv,x,pc_y,proles)
            for role in ('L','U'):
                if role not in wroles:continue
                col=c+(role=='U');slot_pins[col]=(wpins,ppins,role)
            for view,y0 in ((wv,0),(pv,pc_y)):
                for net,y in view.supplies:supply[net].setdefault(y0+y,[]).append(x)
            for role in ('L','U'):
                if role+'B' in wroles:
                    pins[source.instances[wroles[role+'B']]['pins']['A']]=wpins[role+'B.A'][0]
        for col in list(range(cols))+[cols+1]:
            replica=col==cols+1
            sn='XREPLICA_SENSEAMP' if replica else f'XSENSEAMP_{col}'
            sp=add(sa,cx+col*705,sa_y,sn,source.instances[sn]['pins']);sas.append(sp)
            wpins,ppins,role=slot_pins[col]
            for p in ('BL','BLB'):
                tx,ty,_=wpins[role+'.'+p][0]
                if abs(tx-(cx+col*705+(185 if p=='BL' else 520)))>1e-6:raise ValueError('Slot bitline stack off its column')
                wire('m5',tx,ty if split is None else yb,tx,cy)
                ends=[(tx,cy,'m2')]+([(tx,yb,'m2')] if split is not None else [])
                for point in (sp['IN' if p=='BL' else 'INB'],*ends):
                    xx,yy2=lift(point);wire('m4',xx,yy2,tx,yy2);via(4,tx,yy2)
        # Bars run from the left trunks to the rightmost slot on each level.
        last=max(slot_pins)
        for role,net in (('L',control_order[3]),('U',control_order[3])):
            col=max(c for c,(_,_,r) in slot_pins.items() if r==role)
            global_bar(net,slot_pins[col][0][role+'.EN'][0])
        for role in ('L','U'):
            col=max(c for c,(_,_,r) in slot_pins.items() if r==role)
            global_bar(control_order[0],slot_pins[col][1][role+'.ENB'][0])
        # Supply bars reach both ends, whichever side their trunk is on.
        for net,levels in supply.items():
            for y,xs in sorted(levels.items()):
                global_bar(net,(min(xs),y,'m4'));global_bar(net,(max(xs),y,'m4'))
        for ps in sas:
            global_bar('VDD',ps['VDD'])
            x,y,l=ps['VSS'];wire('m2',x,y,x,y-100);global_bar('VSS',(x,y-100,'m2'))
            x,y,l=ps['ISO'];yy=y-300;wire('m3',x,y,x,yy);global_bar(control_order[2],(x,yy,'m3'))
            x,y,l=ps['EN'];yy=y+70;wire('m1',x,y,x,yy);global_bar(control_order[1],vertical_m1((x,yy,'m1')))
    else:
        # Pair write drivers and rotated data buffers in two-column tiles.
        pcs=[];sas=[];wds=[]
        for col in list(range(cols))+[cols+1]:
            replica=col==cols+1
            pn='XPRECHARGE_RBL' if replica else f'XPRECHARGE_{col}'
            sn='XREPLICA_SENSEAMP' if replica else f'XSENSEAMP_{col}'
            wn='XREPLICA_WDRV_LOAD' if replica else f'XWRITEDRIVER_{col}'
            pp=add(pc,cx+col*705,12255,pn,source.instances[pn]['pins']);pcs.append(pp)
            sp=add(sa,cx+col*705,7495,sn,source.instances[sn]['pins']);sas.append(sp)
            pair=col//2;upper=col%2;wx=cx+pair*1410;wy=2000+upper*2705
            wp=add(wd,wx,wy,wn,source.instances[wn]['pins']);wds.append(wp)
            # Each bitline reaches the core on M5. Local M4 jogs clear the two
            # control trunks inside the unchanged write-driver geometry.
            for k,(p,tx,dy) in enumerate([('BL',cx+col*705+185,-170),('BLB',cx+col*705+520,170)]):
                x,y,_=wp[p];yy=y+dy;wire('m3',x,y,x,yy);lift((x,yy,'m3'))
                wire('m4',x,yy,tx,yy);via(4,tx,yy)
                wire('m5',tx,yy,tx,cy)
                for point in (pp[p],sp['IN' if p=='BL' else 'INB'],(tx,cy,'m2')):
                    xx,yy2=lift(point);wire('m4',xx,yy2,tx,yy2);via(4,tx,yy2)
            # DIN uses M6 below its bitline's lower endpoint.
            x,y,_=wp['DIN'];yy=wy+700;wire('m3',x,y,x,yy);lift((x,yy,'m3'))
            tx=wx+(1100 if upper else 400);wire('m4',x,yy,tx,yy);via(4,tx,yy);via(5,tx,yy)
            if not replica:
                bn=f'XDIN_BUF_{col}'
                bp=add(buf,wx+1365,upper*860,bn,source.instances[bn]['pins'],True)
                bx,by=lift(bp['Z']);wire('m4',bx,by,tx,by);via(4,tx,by);via(5,tx,by);wire('m6',tx,by,tx,yy)
                pins[source.instances[bn]['pins']['A']]=bp['A']
            else:
                sx=wp['VSS'][0];via(3,sx,yy);wire('m4',tx,yy,sx,yy)
        # The native repeated-cell rails share supplies/controls. Connect their
        # access points with actual metal; no virtual supplies are used.
        for ps in pcs:
            x,y=lift(ps['VDD'],3);wire('m3',x,y,x,y-180);global_bar('VDD',(x,y-180,'m3'))
            global_bar(control_order[0],ps['ENB'])
        for ps in sas:
            global_bar('VDD',ps['VDD'])
            x,y,l=ps['VSS'];wire('m2',x,y,x,y-100);global_bar('VSS',(x,y-100,'m2'))
            x,y,l=ps['ISO'];yy=y-300;wire('m3',x,y,x,yy);global_bar(control_order[2],(x,yy,'m3'))
            x,y,l=ps['EN'];yy=y+70;wire('m1',x,y,x,yy);global_bar(control_order[1],vertical_m1((x,yy,'m1')))
        for ps in wds:
            for p in ('VDD','VSS'):global_bar(p,ps[p])
            global_bar(control_order[3],ps['EN'])
        # Fill the 45 nm paired-driver seams with the same wells and supply bars.
        for upper in (0,1):
            wy=2000+upper*2705
            for pair in range((cols+2)//2-1):
                x=cx+pair*1410
                for layer,lo,hi in [('pwell',0,372.5),('nwell',372.5,1932.5),('pwell',1932.5,2530)]:
                    b.rect(layer,x+1365,wy+lo,x+1410,wy+hi)
                for phase in (35,660,1485,2200,2530):
                    b.rect('m2',x+1365,wy+phase-35,x+1410,wy+phase+35)
            # Buffer supply accesses are rotated M1 rails.
        for pair in range(cols//2):
            x=cx+pair*1410
            if pair<cols//2-1:b.rect('pwell',x+1365,0,x+1410,1720)
            global_bar('VDD',vertical_m1((x+100,100,'m1')));global_bar('VSS',vertical_m1((x+1265,450,'m1')))
    if include_auxiliary:
        # Replica driver and output latch occupy the space below the decoder.
        # Folded banks keep both blocks below the periphery, on separate
        # levels so their trunk bars never share a height.
        ax=500 if cols==8 else 3500
        ry=-1700. if folded else 480.
        rp=add(rwl,ax,ry,'XRWL',source.instances['XRWL']['pins'])
        lp=add(latch,ax,-3400. if folded else 7000.,'XD_LATCH',source.instances['XD_LATCH']['pins'])
        for p in ('VDD','VSS'):
            global_bar(p,rp[p])
            x,y,l=lp[p];px=250 if cols==8 else 1000
            wire('m1',px,y,x,y);global_bar(p,(px,y,'m1'))
        for p,n,phase in [('A','VDD',985),('B','wl_en',385),('Z','RWL',685)]:
            x,y,l=rp[p];yy=ry+phase;wire('m3',x,y,x,yy);global_bar(n,(x,yy,'m3'))
        global_bar(control_order[1],lp['EN'])
        if cols==8:
            x,y=lift(lp['D'],5);wire('m5',x,y,x,7500);wire('m5',-1000,7500,x,7500);via(5,-1000,7500)
            global_sites.setdefault(control_order[6],[]).append((-1000,7500))
        else:global_bar(control_order[6],lp['D'])
        # Only the selected sense output feeds the source output latch.
        sx,sy,sl=sas[cols-1]['Q'];yy=sy+100;wire('m3',sx,sy,sx,yy)
        global_bar(control_order[6],(sx,yy,'m3'))
    shared={}
    if split is not None:
        # Address buses, wl_en and level-0 enables end at each half's band
        # edge. Bank-only builds join them with straight M5 bridges.
        for net in [n for n in dict.fromkeys([*dp_lo,*dp]) if n not in ('VDD','VSS') and not n.startswith('WL')]:
            ends={k:v[net] for k,v in (('lower',dp_lo),('upper',dp)) if net in v}
            if any(l!='m5' for _,_,l in ends.values()):raise ValueError(f'{net}: decoder band pin is not on M5')
            shared[net]=ends
            if bridge and len(ends)==2:
                (x0,y0,_),(x1,y1,_)=ends['lower'],ends['upper']
                if x0!=x1:raise ValueError(f'{net}: decoder halves disagree on the trunk phase')
                wire('m5',x0,y0,x1,y1)
    for net,sites in global_sites.items():
        xs={x for x,y in sites}
        if len(xs)!=1:raise ValueError('Global trunk phase mismatch')
        x=sites[0][0];ys=sorted(set(y for _,y in sites));clusters=[]
        for yy in ys:
            if clusters and yy-clusters[-1][-1]<310:clusters[-1].append(yy)
            else:clusters.append([yy])
        for cluster in clusters:
            yy=cluster[0]
            wire('m4',x,yy,x,cluster[-1]);via(4,x,yy);via(5,x,yy)
        wire('m6',x,min(ys),x,max(ys))
        pins[net]=(x,sites[0][1],'m6')
    internal={f'WL{r}' for r in range(rows)}|({'RWL'} if include_auxiliary else set())|{source.instances['XDECODER']['pins'][f'WL{r}'] for r in range(rows)}
    ports=tuple(dict.fromkeys([p for p in data_component.ports if p not in internal]+
        [n for name in row_names+aux_names for n in source.instances[name]['pins'].values() if n not in internal]))
    for p in ports:
        if p not in pins:pins[p]=points[p][0]
    pins={p:pins[p] for p in ports}
    for p,(x,y,l) in pins.items():b.label(l,p,x,y)
    comp=Component(b.name,ports,(),tuple(Instance(n,tuple(source.instances[n]['pins'][p] for p in c.ports),c) for n,c in specs.items()))
    bottom=0. if split is None else ylo
    v=CellView(comp,b,cx+core.width_nm,cy+core.height_nm-bottom,pins,terminals,[],[],[])
    v.physical_fingers=fingers;v.placements=placed;v.wordlines=[dict(row=r,start_nm=wl_pins[f'WL{r}'],end_nm=cp[f'WL{r}']) for r in range(rows)]
    from .dense_budget import compare
    # A split bank's halves and register reserve have no OpenRAM reference.
    views=dict(sense=sa,decoder=decoder,core=core) if split is None else dict(sense=sa)
    if not folded:views.update(precharge=pc,write=wd,buffer=buf)
    if include_auxiliary:views.update(latch=latch,**({'replica_driver':rwl} if cols<=32 else {}))
    budgets=compare(views,rows,cols,v if split is None else None)
    metadata=dict(schema=3,artifact_kind='bank',rows=rows,cols=cols,controller_layout=False,
        replica_integrated_in_core=True,included_instances=list(specs),excluded_instances=omitted,
        array_origin_nm=[cx,cy],decoder_bbox_nm=[0,cy if split is None else ylo,decoder.width_nm,cy+decoder.height_nm],
        periphery_bands_nm=(dict(write=[0,wslot.height],sense=[sa_y,sa_y+sa.height_nm],precharge=[pc_y,pc_y+pslot.height],
            slot_pitch=1410,slots=cols//2+1,write_supplies=[list(t) for t in wslot.supplies]) if folded else None),
        core_geometry=dict(width_nm=core.width_nm,height_nm=core.height_nm,replica_x_nm=(cols+1)*705),
        external_pins=pins,area_comparison=budgets,waveform_validation='not run',defaults_promotable=False)
    if split is not None:
        metadata['split']=dict(row=split,lower_rows=[0,split],upper_rows=[split,rows],
            lower_array_origin_nm=[cx,ylo],lower_array_top_nm=yb,register_reserve_nm=reserve,
            upper_decoder_bbox_nm=[0,cy,decoder.width_nm,cy+decoder.height_nm],
            lower_decoder_bbox_nm=[0,ylo,decoder_lo.width_nm,yb],
            decoder_band_pins=shared,decoder_bridges=bridge,decoder_spare_nm=spare,
            decoder_row_left_nm={**decoder_lo.row_left_nm,**decoder.row_left_nm},
            decoder_row_metal_nm={**decoder_lo.row_metal_nm,**decoder.row_metal_nm},
            decoder_row_well_nm={**decoder_lo.row_well_nm,**decoder.row_well_nm},
            bitlines='continuous M5 from the lower array top edge through the periphery to the upper array')
    summary=Path(deck).with_name('summary.json')
    if summary.is_file():
        import json,hashlib
        data=json.loads(summary.read_text())
        metadata['compiler_provenance']={k:data.get(k) for k in ('compiler_version','corner','vdd','temperature',
            'variation_mode','real_cell_mode','seed','model_sha256','interconnect','timing','driver_sizes')}
        metadata['compiler_summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
    scope='complete bank excluding TIME_CONTROL' if include_auxiliary else 'array, decoder and column periphery; TIME_CONTROL, RWL driver and output latch external'
    return write_view(v,output,deck,scope,metadata)
