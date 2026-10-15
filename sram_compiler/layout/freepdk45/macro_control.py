"""Integrate TIME_CONTROL beside the decoder with planned, search-free routing.

The floorplan follows OpenRAM's sram_1bank and sram22's channel discipline:

* Control logic sits left of the decoder. Every bank-facing pin is on its
  right edge at the height of the bank conductor it drives or observes.
* The 32 data registers form a pitch-planned array below the column
  periphery (``control_registers.py``) instead of a block inside the
  controller frame.
* Each connection is an explicit one- to three-segment route on an assigned
  layer, and the data bus uses a left-edge channel with vertical constraints.

Layer roles: M3/M4 data channel; M5 risers and M6 straight drives into the
bank's control trunks; M7/M8 far-end and replica returns; M9 supply straps;
M10 supply and register-clock drops. A layout-level spacing check and
metal/poly connectivity audits run before export; official DRC/LVS remains
the acceptance check.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json
from .components import Component,Instance,CellView,read_component
from .control_registers import make_data_registers,ROWS
from .full_source import flatten
from .dense_views import write_view
from .dense_aux import make_aux
from .lib import Builder,new_layout,pya,u
from .time_control import make_control,units

GRID=100.
LOGIC_WIDTH=15300.     # widest frozen driver plus the RC-guard column
LOGIC_GAP=1000.        # controller right edge to the decoder's left edge
PAD_Y=1000.            # M8 supply/clock pads above the first RC guard
PADS=dict(VSS=900.,VDD=2600.,clk_buf=4300.)
DATA_TOP_TRACK=-700.   # highest data-channel M4 track, below the DIN escapes
TRACK=280.
STAGGER=600.           # via7 stacks on adjacent 805 nm register tracks
ARRAY_MARGIN=600.      # channel via7 to the top register's own Q via7 >= 0.44 um
STRAP=dict(width=1000.,space=1000.)   # wide-M9 spacing rule is 0.9 um
# Wires match their via landings, so stacks leave no 2.5 nm steps.
WIDTH={'m1':70.,'m2':70.,'m3':70.,'m4':140.,'m5':145.,'m6':145.,'m7':400.,'m8':405.,'m9':805.,'m10':805.}
MINW={'m1':65.,'m2':70.,'m3':70.,'m4':140.,'m5':140.,'m6':140.,'m7':400.,'m8':400.,'m9':800.,'m10':800.}
SPACE={'m1':65.,'m2':70.,'m3':70.,'m4':140.,'m5':140.,'m6':140.,'m7':400.,'m8':400.,'m9':800.,'m10':800.}
# Address and enable pins share M6 heights below sa_iso; the decoder pins
# are reached by M5 risers, so any order is crossing-free.
ADDRESS_Y=dict(wl_en=6000.,A_dff0=6500.,A_dff1=7000.,A_dff2=7500.,A_dff3=8000.,A_dff4=8500.)
# Far-end returns use M7 tracks at least 1 um apart. sen/iso/wen run at
# their bar heights; PRE jogs up at its tap, beyond the s_en tap.
RETURN_Y=dict(pre_far=12900.,sen_far=11900.,iso_far=9400.,wen_far=5100.,rbl=7300.,rwl=15500.)
FEEDBACK={'pre_far':'PRE','iso_far':'sa_iso','sen_far':'s_en','wen_far':'w_en','rwl':'RWL','rbl':'RBL'}
DRIVES=('PRE','s_en','sa_iso','w_en')
WRITE_ROW_PITCH=2705.  # dense_bank pairs write drivers in two rows; w_en has a bar in each
REPLICA_TAP_Y=7300.
RWL_DROP_X=600.        # M8 return over the decoder's left edge
VIA={1:65.,2:70.,3:70.,4:140.,5:140.,6:140.,7:400.,8:400.,9:800.}
LANDING={4:140.,5:145.,6:145.,7:400.,8:405.,9:805.,10:805.}


def snap(v):return round(v/GRID)*GRID


_conductors={}
def bank_conductor(cell,layer,x,y):
    """Bounding box (nm) of the merged bank conductor containing a point."""
    key=cell.cell_index(),layer
    if key not in _conductors:
        _conductors[key]=pya.Region(cell.begin_shapes_rec(cell.layout().layer(*{'m4':(17,0),'m5':(19,0),'m6':(21,0)}[layer]))).merged()
    hits=[pg.bbox() for pg in _conductors[key].each() if pg.inside(pya.Point(u(x),u(y)))]
    if len(hits)!=1:raise ValueError(f'No unique bank {layer} conductor at {x},{y}')
    s=cell.layout().dbu*1000;b=hits[0]
    return b.left*s,b.bottom*s,b.right*s,b.top*s


def pin_plan(bm,width=LOGIC_WIDTH):
    """Controller pins in its own frame; its bottom is at bank y=0."""
    bp=bm['external_pins'];e6=width-100;e7=width-300
    plan={n:(e6,snap(bp[n][1]),'m6') for n in DRIVES if n!='w_en'}
    # w_en lands on its two-row M6 trunk; take the trunk middle.
    plan['w_en']=(e6,snap(bp['w_en'][1]+WRITE_ROW_PITCH/2),'m6')
    plan.update({n:(e6,y,'m6') for n,y in ADDRESS_Y.items()})
    plan.update({n:(e7,y,'m7') for n,y in RETURN_Y.items()})
    plan.update({n:(x,PAD_Y,'m8') for n,x in PADS.items()})
    ys=sorted(p[1] for n,p in plan.items() if p[2]=='m6')
    if min(b-a for a,b in zip(ys,ys[1:]))<300:raise ValueError('Controller M6 edge pins are closer than one track')
    return plan


def channel_tracks(nets,pitch=140.):
    """OpenRAM-style left-edge channel assignment with vertical constraints.

    nets maps a name to (bottom x, top x); track 0 is the lowest. A bottom
    pin within one M3 pitch of another net's top pin must use a lower track.
    """
    below={n:set() for n in nets}
    for a,(xb,_) in nets.items():
        for b,(xbb,xt) in nets.items():
            if a==b:continue
            if abs(xb-xbb)<pitch:raise ValueError(f'Data channel bottom pins collide: {a}, {b}')
            if abs(xb-xt)<pitch:below[b].add(a)
    result={};track=0
    while len(result)<len(nets):
        placed=set(result);right=None;chosen=[]
        for n in sorted((n for n in nets if n not in placed and below[n]<=placed),key=lambda n:min(nets[n])):
            lo,hi=sorted(nets[n])
            if right is None or lo-70>=right+140:chosen.append(n);right=hi+70
        if not chosen:raise ValueError('Cyclic data-channel vertical constraints')
        result.update((n,track) for n in chosen);track+=1
    return result


class Planner:
    """Explicit wires and via stacks; every macro-level shape is recorded."""
    def __init__(self,builder):
        self.b=builder;self.drawn=defaultdict(pya.Region);self.segments=[]
    def rect(self,layer,x0,y0,x1,y1):
        self.b.rect(layer,x0,y0,x1,y1)
        self.drawn[layer].insert(pya.Box(u(min(x0,x1)),u(min(y0,y1)),u(max(x0,x1)),u(max(y0,y1))))
    def wire(self,net,layer,pts,width=None):
        w=(width or WIDTH[layer])/2
        for (x,y),(xx,yy) in zip(pts,pts[1:]):
            if x!=xx and y!=yy:raise ValueError(f'{net}: diagonal {layer} segment')
            self.rect(layer,min(x,xx)-w,min(y,yy)-w,max(x,xx)+w,max(y,yy)+w)
            self.segments.append((net,layer,(x,y),(xx,yy)))
    def stack(self,net,x,y,lo,hi):
        # The cuts and landings of the verified TIME_CONTROL integration.
        for n in range(lo,hi):
            h=VIA[n]/2;self.rect(f'v{n}',x-h,y-h,x+h,y+h)
            for k in (n,n+1):
                if k==1:self.rect('m1',x-67.5,y-35,x+67.5,y+35)
                elif k<4:self.rect(f'm{k}',x-35,y-70,x+35,y+70)
                else:h=LANDING[k]/2;self.rect(f'm{k}',x-h,y-h,x+h,y+h)
        self.segments.append((net,'via',(x,y,lo),(x,y,hi)))
    def check(self,cell):
        """Width/space violations touching planned shapes (Calibre is final)."""
        errors=[];ly=cell.layout()
        for layer,space in SPACE.items():
            mine=self.drawn[layer].merged()
            if mine.is_empty():continue
            full=pya.Region(cell.begin_shapes_rec(ly.layer(*{f'm{i}':(9+2*i,0) for i in range(1,11)}[layer]))).merged()
            near=mine.sized(u(space))
            for pairs in (full.space_check(u(space)),full.width_check(u(MINW[layer]))):
                for pair in pairs.each():
                    if not (pya.Region(pair.bbox()) & near).is_empty():
                        b=pair.bbox();s=ly.dbu*1000
                        errors.append(f'{layer} {b.left*s:.1f},{b.bottom*s:.1f} {b.right*s:.1f},{b.top*s:.1f}')
        if errors:raise ValueError('Planned macro routes violate spacing:\n'+'\n'.join(errors[:40]))


def logic_manifest(deck,v,plan,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ly=new_layout();top=ly.create_cell(v.builder.name);top.copy_tree(v.builder.cell)
    ly.write(str(output/'full_macro.gds'))
    m=dict(artifact_kind='controller_partition',partition='control_logic',primary=v.builder.name,
           excluded_units=['Xdff_buf_data/'],width_nm=v.width_nm,height_nm=v.height_nm,pins=v.pins,pin_plan=plan,
           device_pins=v.device_pins,physical_fingers=v.physical_fingers,placements=v.placements,
           passive_locations=v.passive_locations,standard_cell_inventory=v.standard_cell_inventory,
           source_deck_sha256=hashlib.sha256(Path(deck).read_bytes()).hexdigest(),
           layout_sha256=hashlib.sha256((output/'full_macro.gds').read_bytes()).hexdigest(),
           generator_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (output/'layout.json').write_text(json.dumps(m,indent=2)+'\n')
    return m


def build_logic(deck,output,bank_dir):
    """Control logic with edge pins planned from a source-matched bank."""
    bm=json.loads((Path(bank_dir)/'layout.json').read_text())
    tc=read_component(deck,'XTIME_CONTROL',allow_passives=True)
    plan=pin_plan(bm)
    v=make_control(new_layout(),tc,LOGIC_WIDTH,exclude=('Xdff_buf_data/',),pin_plan=plan)
    return logic_manifest(deck,v,plan,output)


def build(deck,output,bank_dir,control_dir):
    source=flatten(deck);rows,cols=source.rows,source.cols
    if (rows,cols)!=(32,32):raise ValueError('The controller integration currently supports the 32x32 physical plan')
    latch=source.instances['XD_LATCH']['pins'];sense=source.instances[f'XSENSEAMP_{cols-1}']['pins']
    if latch['D']!=sense['Q'] or latch['EN']!=sense['EN']:
        raise ValueError('The controller plan requires the last-column sense output and enable')
    bank_dir,control_dir=Path(bank_dir),Path(control_dir)
    bm=json.loads((bank_dir/'layout.json').read_text());cm=json.loads((control_dir/'layout.json').read_text())
    digest=hashlib.sha256(Path(deck).read_bytes()).hexdigest()
    if any(m['source_deck_sha256']!=digest for m in (bm,cm)):raise ValueError('Physical blocks do not match the source deck')
    for directory,manifest in ((bank_dir,bm),(control_dir,cm)):
        if hashlib.sha256((directory/'full_macro.gds').read_bytes()).hexdigest()!=manifest['layout_sha256']:
            raise ValueError('Physical block GDS differs from its layout manifest')
    if set(bm.get('excluded_instances',[]))!={'XTIME_CONTROL','XRWL','XD_LATCH'}:raise ValueError('Controller integration requires the bank with auxiliary blocks external')
    if cm.get('partition')!='control_logic' or cm['pin_plan']!={k:list(v) for k,v in pin_plan(bm,cm['width_nm']).items()}:
        raise ValueError('Expected control logic generated for this bank pin plan')
    ly=new_layout();b=Builder(ly,f'OPENYIELD_CONTROLLED_{rows}X{cols}');P=Planner(b)
    def imported(directory,m):
        src=new_layout();src.read(str(directory/'full_macro.gds'))
        cell=Builder(ly,m['primary']);cell.cell.copy_tree(src.cell(m['primary']));return cell
    bank=imported(bank_dir,bm);control=imported(control_dir,cm)
    bp=bm['external_pins'];cx,cy=bm['array_origin_nm']
    xr=-LOGIC_GAP;xl=xr-cm['width_nm']
    b.place(bank);b.place(control,xl,0)
    cp={n:(x+xl,y,l) for n,(x,y,l) in cm['pins'].items()}
    cb=control.cell.dbbox();controller_box=[cb.left*1000+xl,cb.bottom*1000,cb.right*1000+xl,cb.top*1000]
    endpoints=defaultdict(list);links=[]
    def endpoint(net,point,alias=None):endpoints[net].append((point,alias or net))
    # 1. Drives: one straight M6 line into each bank control trunk.
    for net in DRIVES:
        x,y,_=cp[net];tx=bp[net][0]
        _,by0,_,by1=bank_conductor(bank.cell,'m6',tx,bp[net][1]+(WRITE_ROW_PITCH/2 if net=='w_en' else 0))
        if min(y+72.5,by1)-max(y-72.5,by0)<70:raise ValueError(f'{net} drive misses its trunk')
        P.wire(net,'m6',[(x,y),(tx,y)])
        endpoint(net,(x,y,'m6'));endpoint(net,(tx,bp[net][1],'m6'))
        links.append(dict(controller_port=net,bank_net=net,far_end_return=False,start=cp[net],bank_access=[tx,y,'m6'],route='M6 straight'))
    # 2. Address/enable: M6 to the decoder pin column, M5 riser into the pin.
    for net in ADDRESS_Y:
        x,y,_=cp[net];px,py,pl=bp[net]
        if pl!='m5' or py<y:raise ValueError(f'{net} decoder pin is not an M5 access above the controller pin')
        P.wire(net,'m6',[(x,y),(px,y)]);P.stack(net,px,y,5,6);P.wire(net,'m5',[(px,y),(px,py)])
        endpoint(net,(x,y,'m6'));endpoint(net,(px,py,'m5'))
        links.append(dict(controller_port=net,bank_net=net,far_end_return=False,start=cp[net],bank_access=[px,py,'m5'],route='M6 + M5 riser'))
    # 3. Far-end returns: bar extension to a tap right of the bank's supply
    # trunks, then M7. PRE taps one column further out, beyond the s_en tap.
    tap=snap(bp['VSS'][0]+72.5+140+72.5+GRID/2)
    tap_x=dict(sen_far=tap,iso_far=tap,wen_far=tap,pre_far=tap+900)
    bars={}
    for p,net in FEEDBACK.items():
        if p in ('rwl','rbl'):continue
        trunk=bp[net]
        y=trunk[1]+(WRITE_ROW_PITCH if net=='w_en' else 0)   # w_en: upper write-driver bar
        _,b0,b1,t0=bank_conductor(bank.cell,'m4',cx+cols*705-200,y)
        if not b0<y<t0:raise ValueError(f'{net} bar not found at its trunk height')
        bars[p]=(b1,(b0+t0)/2)
    for p,(end,y) in bars.items():
        net=FEEDBACK[p];tx=tap_x[p];x,ry,_=cp[p]
        P.wire(net,'m4',[(end-70,y),(tx,y)],140.);P.stack(net,tx,y,4,7)
        path=[(x,ry),(tx,ry)] if abs(ry-y)<=30 else [(x,ry),(tx,ry),(tx,y)]
        P.wire('RETURN_'+p,'m7',path)
        endpoint('RETURN_'+p,(x,ry,'m7'),net);endpoint('RETURN_'+p,(end-100,y,'m4'),net)
        links.append(dict(controller_port=p,bank_net=net,far_end_return=True,start=cp[p],bank_access=[tx,y,'m4'],route='M4 bar end + M7'))
    # Replica bitline: tap its M5 column conductor and return on M7.
    rbx,rby,rbl=bp['RBL'];bank_conductor(bank.cell,'m5',rbx,REPLICA_TAP_Y)
    x,ry,_=cp['rbl'];P.stack('RBL',rbx,REPLICA_TAP_Y,5,7);P.wire('RETURN_rbl','m7',[(x,ry),(rbx,ry)])
    endpoint('RETURN_rbl',(x,ry,'m7'),'RBL');endpoint('RETURN_rbl',(rbx,REPLICA_TAP_Y,'m5'),'RBL')
    links.append(dict(controller_port='rbl',bank_net='RBL',far_end_return=True,start=cp['rbl'],bank_access=[rbx,REPLICA_TAP_Y,'m5'],route='M5 replica column + M7'))
    # Replica wordline: observe the active replica row at the array's far
    # end; M7 above the array, M8 down over the decoder, M7 into the pin.
    rwl_tap=(cx+(cols+1)*705+500,cy+(rows-.5)*1365)
    bank_conductor(bank.cell,'m4',*rwl_tap)
    x,ry,_=cp['rwl'];drop=RWL_DROP_X
    P.stack('RWL',*rwl_tap,4,7);P.wire('RETURN_rwl','m7',[rwl_tap,(drop,rwl_tap[1])])
    P.stack('RWL',drop,rwl_tap[1],7,8);P.wire('RETURN_rwl','m8',[(drop,rwl_tap[1]),(drop,ry)])
    P.stack('RWL',drop,ry,7,8);P.wire('RETURN_rwl','m7',[(drop,ry),(x,ry)])
    endpoint('RETURN_rwl',(x,ry,'m7'),'RWL');endpoint('RETURN_rwl',(*rwl_tap,'m4'),'RWL')
    links.append(dict(controller_port='rwl',bank_net='RWL',far_end_return=True,start=cp['rwl'],bank_access=[*rwl_tap,'m4'],route='M7 + M8 + M7'))
    # 4. Data registers: pitch-planned array and a left-edge channel.
    tc=read_component(deck,'XTIME_CONTROL',allow_passives=True)
    registers=[(p,c,m) for p,c,m in units(tc) if p.startswith('Xdff_buf_data/')]
    escapes={}
    for bit in range(cols):
        x,y,l=bp[f'DIN_dff{bit}']
        escapes[f'DIN_dff{bit}']=(x-322.5 if y>600 else x+77.5,x,y)
    order=sorted(escapes,key=lambda n:escapes[n][0])
    by_q={m['Q']:p for p,c,m in registers}
    slots={by_q[n]:(k//ROWS,ROWS-1-k%ROWS) for k,n in enumerate(order)}
    clock_x=xl+PADS['clk_buf']
    # Right-align the register array under the column periphery, leaving the
    # bank supply trunks free to continue down beside it.
    trunk_left=min(bp['VDD'][0],bp['VSS'][0])-72.5
    probe=make_data_registers(new_layout(),registers,slots,top_margin=ARRAY_MARGIN)
    x0=snap(trunk_left-500-probe.width_nm-300)
    nets={n:(x0+probe.pins['Q:'+n][0],escapes[n][0]) for n in order}
    tracks=channel_tracks(nets);count=max(tracks.values())+1
    track_y={t:DATA_TOP_TRACK-TRACK*(count-1-t) for t in range(count)}
    stack_y=track_y[0]-300-STAGGER;array_top=stack_y-ARRAY_MARGIN;y0=array_top-probe.height_nm
    regs=make_data_registers(ly,registers,slots,clock_x=clock_x-x0,top_margin=ARRAY_MARGIN)
    b.place(regs.builder,x0,y0)
    rp={n:(x+x0,y+y0,l) for n,(x,y,l) in regs.pins.items()}
    for n in order:
        qx,qy,_=rp['Q:'+n];ex,px,py=escapes[n];ty=track_y[tracks[n]]
        if qy!=stack_y:raise ValueError('Register Q tracks do not reach the channel stack height')
        if py>600:P.wire(n,'m3',[(px,py),(ex,py),(ex,ty)])
        else:
            P.wire(n,'m3',[(px,py),(px,-400.),(ex,-400.),(ex,ty)]);P.rect('m3',min(px,ex)-35,-470,max(px,ex)+35,-330)
        # Neighbouring 805 nm tracks alternate stack heights for via7 spacing.
        sy=qy+(STAGGER if slots[by_q[n]][1]==2 else 0)
        if sy!=qy:P.wire(n,'m8',[(qx,qy),(qx,sy)])
        P.stack(n,qx,sy,3,8);P.wire(n,'m3',[(qx,sy),(qx,ty)])
        if abs(qx-ex)<140:
            # A track within one M3 pitch of its own escape joins on M3.
            P.rect('m3',min(qx,ex)-35,ty-35,max(qx,ex)+35,ty+35)
        else:
            P.stack(n,qx,ty,3,4);P.stack(n,ex,ty,3,4);P.wire(n,'m4',[(qx,ty),(ex,ty)],140.)
        endpoint(n,(qx,qy,'m8'));endpoint(n,(px,py,'m3'))
    # Register clock: controller M8 pad, M10 drop to the M5 trunk extended
    # just above the array, clear of the supply straps.
    kx,ky,_=rp['CLK'];px,py,_=cp['clk_buf']
    if abs(kx-px)>1e-6:raise ValueError('Register clock trunk is not under the controller clock pad')
    clock_y=array_top+300;P.wire('clk_buf','m5',[(kx,ky),(kx,clock_y)])
    P.stack('clk_buf',px,py,8,10);P.wire('clk_buf','m10',[(px,py),(kx,clock_y)]);P.stack('clk_buf',kx,clock_y,5,10)
    endpoint('clk_buf',(px,py,'m8'));endpoint('clk_buf',(kx,ky,'m5'))
    # 5. Supplies: two M9 straps over the register array, between its D/Q
    # via rows, keep the data channel to the data bus. M10 drops from the
    # controller pads; stacks reach the register trunks and bank trunks.
    rail_rows=sorted({y for k,(x,y,l) in rp.items() if k.startswith('D:')})
    straps=dict(VSS=(rail_rows[0]+rail_rows[1])/2-50,VDD=(rail_rows[1]+rail_rows[2])/2+50)
    low,high=sorted(straps.values())
    if high-low-STRAP['width']<STRAP['space']:raise ValueError('Supply straps violate wide-M9 spacing')
    if any(abs(sy-ry)<STRAP['width']/2+202.5+150 for sy in straps.values() for ry in rail_rows):
        raise ValueError('Supply straps cover register D/Q via rows')
    right={'VDD':bp['VDD'][0],'VSS':bp['VSS'][0]}
    for net,sy in straps.items():
        px,py,_=cp[net];P.stack(net,px,py,8,10);P.wire(net,'m10',[(px,py),(px,sy)]);P.stack(net,px,sy,9,10)
        P.wire(net,'m9',[(px-400,sy),(right[net]+400,sy)],STRAP['width'])
        # Bank trunks continue down past the register rows on M6.
        tx,ty,_=bp[net];P.wire(net,'m6',[(tx,ty),(tx,y0)]);P.stack(net,tx,sy,6,9)
        gx,gy,_=rp[net];P.stack(net,gx,sy,3,9)
        endpoint(net,(px,py,'m8'));endpoint(net,(tx,ty,'m6'));endpoint(net,(gx,gy,'m3'))
    for (net,r),(x,y,l) in regs.row_rails.items():
        P.wire(net,'m2',[(x+x0,y+y0),(right[net],y+y0)]);P.stack(net,right[net],y+y0,2,6)
    # 6. Replica wordline driver above the decoder and the output latch.
    auxiliaries={};auxiliary_positions={};auxiliary_pins={}
    for name,x,y in [('XRWL',bm['decoder_bbox_nm'][2]-2380,cy+rows*1365+120),
                     ('XD_LATCH',tap+300,bp[f'SA_Q{cols-1}'][1]-200)]:
        c=read_component(deck,name);v=make_aux(ly,c);auxiliaries[name]=v
        auxiliary_positions[name]=(x,y);b.place(v.builder,x,y)
        auxiliary_pins[name]={p:(px+x,py+y,l) for p,(px,py,l) in v.pins.items()}
    rp_=auxiliary_pins['XRWL'];lp=auxiliary_pins['XD_LATCH']
    rpx,rpy=auxiliary_positions['XRWL'];lpx,lpy=auxiliary_positions['XD_LATCH']
    b.rect('pwell',rpx,cy+rows*1365,rpx+auxiliaries['XRWL'].width_nm,rpy)
    ax,ay,_=rp_['A'];vx,vy,_=rp_['VDD']
    P.wire('VDD','m3',[(ax,ay),(ax,rpy+1100),(vx,rpy+1100)]);P.stack('VDD',vx,vy,1,3);P.wire('VDD','m3',[(vx,rpy+1100),(vx,vy)])
    zx,zy,_=rp_['Z']
    if abs(zy-(cy+(rows+.5)*1365))>1e-6:raise ValueError('Replica-driver output is not aligned to the core RWL')
    P.stack('RWL',zx,zy,3,6)
    # Its enable joins the far (top) end of the decoder's wl_en M5 spine.
    bx,by,_=rp_['B'];spine=(bp['wl_en'][0],cy+(rows-1)*1365+80)
    bank_conductor(bank.cell,'m5',*spine)
    P.wire('wl_en','m3',[(bx,by),(bx,rpy+385)]);P.stack('wl_en',bx,rpy+385,3,4)
    P.wire('wl_en','m4',[(bx,rpy+385),(spine[0],rpy+385)]);P.stack('wl_en',spine[0],rpy+385,4,5)
    P.wire('wl_en','m5',[(spine[0],rpy+385),spine])
    endpoint('wl_en',(bx,by,'m3'))
    # Supplies extend the decoder's left-edge M2 rails up to its M1 rails.
    for net,rail_x,top_y in (('VSS',200.,cy+(rows-1)*1365+100),('VDD',50.,cy+(rows-1)*1365+1265)):
        px,py,_=rp_[net]
        P.wire(net,'m1',[(px,py),(rail_x,py)]);P.stack(net,rail_x,py,1,2);P.wire(net,'m2',[(rail_x,top_y),(rail_x,py)])
        endpoint(net,(px,py,'m1'))
    # The latch D track continues the selected sense output on M4.
    qx,qy,_=bp[f'SA_Q{cols-1}'];dx,dy,_=lp['D']
    if abs(dy-qy)>1e-6:raise ValueError('Latch D track is not aligned to the selected sense output')
    P.stack('LATCH_DATA',qx,qy,3,4);P.wire('LATCH_DATA','m4',[(qx,qy),(dx,dy)],140.)
    ex,ey,_=lp['EN'];sx=tap_x['sen_far'];sy=bars['sen_far'][1]
    P.wire('LATCH_ENABLE','m5',[(sx,sy),(ex,sy),(ex,ey)]);P.stack('LATCH_ENABLE',ex,ey,4,5)
    endpoint('LATCH_DATA',(qx,qy,'m3'),latch['D']);endpoint('LATCH_DATA',(dx,dy,'m4'),latch['D'])
    endpoint('LATCH_ENABLE',(ex,ey,'m4'),latch['EN'])
    for net,ty,tx in (('VSS',lpy-270,right['VSS']),('VDD',lpy+630,right['VDD'])):
        px,py,_=lp[net];side=lpx-200
        P.wire(net,'m1',[(px,py),(side,py),(side,ty),(tx,ty)]);P.stack(net,tx,ty,1,6)
        endpoint(net,(px,py,'m1'))
    P.check(b.cell)
    # Audit actual metal/poly conductors: no cross-net shorts before export,
    # and after drawing every endpoint of a net is one conductor.
    from .conductor_ownership import port_conductors
    check_ports={};check_aliases={}
    for net,points in endpoints.items():
        for i,(point,alias) in enumerate(points):
            key=f'{net}:{i}';check_ports[key]=point;check_aliases[key]=alias.upper()
    port_conductors(b,check_ports,check_aliases,require_connected=True)
    ports=tuple(source.ports);pins={}
    for p in ports:
        if p in ('VDD','VSS'):pins[p]=(cp[p][0],straps[p],'m9')
        elif p in bp:pins[p]=bp[p]
        elif p in ('OUT','OUT_B'):pins[p]=lp['Q' if p=='OUT' else 'QB']
        elif p.startswith('DIN') and 'D:'+p in rp:pins[p]=rp['D:'+p]
        else:pins[p]=cp[p]
        x,y,l=pins[p];b.label(l,p,x,y)
    formal_map=source.instances['XTIME_CONTROL']['pins']
    tc_ports=tc.ports;bank_ports=tuple(bm['pins'])
    specs={n:read_component(deck,n) for n in bm['included_instances']}
    bc=Component(bm['primary'],bank_ports,(),tuple(Instance(n,tuple(source.instances[n]['pins'][p] for p in c.ports),c) for n,c in specs.items()))
    component=Component(b.name,ports,(),(Instance('XBANK',bank_ports,bc),Instance('XTIME_CONTROL',tuple(formal_map[p] for p in tc_ports),tc),
        *(Instance(n,tuple(source.instances[n]['pins'][p] for p in v.component.ports),v.component) for n,v in auxiliaries.items())))
    from .dense_views import expand
    bank_pins,bank_fingers=expand(bm)
    terminals={'XBANK/'+n:tuple(p) for n,p in bank_pins.items()}
    terminals.update({'XTIME_CONTROL/'+n:(x+xl,y,l) for n,(x,y,l) in cm['device_pins'].items()})
    terminals.update({'XTIME_CONTROL/'+n:(x+x0,y+y0,l) for n,(x,y,l) in regs.device_pins.items()})
    physical=[]
    def shifted(prefix,fs,dx,dy):
        for f in fs:
            f=dict(f,device=prefix+'/'+f['device'])
            if 'gate_xy_nm' in f:f['gate_xy_nm']=(f['gate_xy_nm'][0]+dx,f['gate_xy_nm'][1]+dy)
            physical.append(f)
    shifted('XBANK',bank_fingers,0,0);shifted('XTIME_CONTROL',cm['physical_fingers'],xl,0)
    shifted('XTIME_CONTROL',regs.physical_fingers,x0,y0)
    for name,av in auxiliaries.items():
        dx,dy=auxiliary_positions[name]
        terminals.update({name+'/'+n:(x+dx,y+dy,l) for n,(x,y,l) in av.device_pins.items()})
        shifted(name,av.physical_fingers,dx,dy)
    box=b.cell.dbbox();v=CellView(component,b,box.width()*1000,box.height()*1000,pins,terminals,[],P.segments,[])
    v.physical_fingers=physical
    metadata=dict(artifact_kind='bank',rows=rows,cols=cols,controller_layout=True,
        included_instances=list(bm['included_instances'])+['XTIME_CONTROL','XRWL','XD_LATCH'],excluded_instances=[],
        routing='planned tracks; no maze search or negotiation',
        controller_bbox_nm=controller_box,controller_frame_nm=[xl,0.,xr,cm['height_nm']],controller_origin_nm=[xl,0.],controller_area_um2=cb.area(),
        controller_pin_plan=cm['pin_plan'],controller_standard_cells=cm['standard_cell_inventory'],
        data_register_bbox_nm=[x0,y0,x0+regs.width_nm,y0+regs.height_nm],data_register_slots={by_q[n].split('/')[-1]+'->'+n:list(slots[by_q[n]]) for n in order},
        data_channel=dict(layers=['m3','m4'],tracks=count,track_pitch_nm=TRACK,track_y_nm=[track_y[t] for t in range(count)],
            nets={n:dict(register_x_nm=nets[n][0],escape_x_nm=nets[n][1],track=tracks[n]) for n in order}),
        supply_straps_nm={n:dict(layer='m9',y=y,width=1000.) for n,y in straps.items()},
        controller_links=links,auxiliary_positions_nm=auxiliary_positions,
        array_origin_nm=bm['array_origin_nm'],decoder_bbox_nm=bm['decoder_bbox_nm'],core_geometry=bm['core_geometry'],
        replica_integrated_in_core=True,source_devices=source.mos,source_passives=source.passives,
        source_artifacts={str(d):hashlib.sha256((d/'full_macro.gds').read_bytes()).hexdigest() for d in (bank_dir,control_dir)},
        passive_locations=cm['passive_locations'],waveform_validation='not run',defaults_promotable=False,
        feedback_max_metal=8,global_routing_grid_nm=GRID,area_limits_um2=dict(controller_with_registers=872.20185,full_macro=8811.4131),
        external_pins=pins,compiler_provenance=bm.get('compiler_provenance'))
    if box.area()>8811.4131+1e-9:raise ValueError('Complete macro exceeds the recorded OpenRAM full-macro area')
    return write_view(v,output,deck,'complete bank with TIME_CONTROL and input registers',metadata)


def generate(deck,output,controller_layout_dir=None):
    """Generate source-matched parts and the controlled bank in fresh directories."""
    from .dense_bank import build as build_bank
    output=Path(output);parts=output.with_name(output.name+'_parts')
    source=flatten(deck)
    if (source.rows,source.cols)!=(32,32):
        raise ValueError('The controller integration currently supports 32x32 nominal source decks')
    bank_dir=parts/'bank'
    build_bank(deck,bank_dir,include_auxiliary=False)
    if controller_layout_dir is None:
        controller_layout_dir=parts/'control_logic'
        build_logic(deck,controller_layout_dir,bank_dir)
    return build(deck,output,bank_dir,controller_layout_dir)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--controller-layout-dir',type=Path,
                        help='reuse control logic generated from exactly this source deck and bank pin plan')
    args=parser.parse_args(argv)
    generate(args.deck,args.output_dir,args.controller_layout_dir)
    print(args.output_dir/'full_macro.gds')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
