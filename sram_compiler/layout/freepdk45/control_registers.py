"""Planned data-register array below the column periphery.

The 32 source data registers keep their frozen transmission-gate DFF master.
They leave the TIME_CONTROL frame and form four abutted register rows, as in
OpenRAM's separate data DFF array. Each register's Q leaves the array top on
its own M8 track: rows below the top jog on M7 to a unique track and cross the
registers above, whose own M8 use is limited to their Q and D landings. Rows
share wells and M1/M2 supplies; the CLK M6 rails join by abutment and meet a
left-side M5 clock trunk. No search routing is used.
"""
from .components import CellView
from .control_register import make_register
from .lib import Builder,new_layout,pya,u

ROWS=4
COLUMNS=8
GAP=190.
ROW_PITCH=1900.
TRACK_PITCH=805.   # 400 nm M8 track beside a 405 nm via7 landing, 400 nm apart


def make_data_registers(layout,registers,slots,clock_x=-1100.,top_margin=500.):
    """Place DFF instances on a ROWS x COLUMNS grid.

    registers: [(path, component, mapping)] for the source data DFFs.
    slots: path -> (column, row), with row 0 at the top of the array.
    """
    if len(registers)!=ROWS*COLUMNS or sorted(slots.values())!=sorted((j,r) for j in range(COLUMNS) for r in range(ROWS)):
        raise ValueError('Data-register array needs one register in every planned slot')
    component=registers[0][1]
    if any(c.identity!=component.identity for _,c,_ in registers):raise ValueError('Data registers must share one source DFF')
    view=make_register(new_layout(),component)
    master=layout.cell(view.builder.name)
    if master is None:
        master=layout.create_cell(view.builder.name);master.copy_tree(view.builder.cell)
    b=Builder(layout,'TIME_CONTROL_DATA_REGISTERS_'+component.identity[:12])
    w,h=view.width_nm,view.height_nm;pitch=w+GAP
    xd,yd,_=view.pins['D'];xq,yq,_=view.pins['Q']
    # Rows below the top cross the registers above on distinct M8 tracks.
    tracks=[xq]+[xq-995.-TRACK_PITCH*(r-1) for r in range(1,ROWS)]
    if tracks[-1]-200<xd+202.5+400:raise ValueError('Lowest Q track violates M7/M8 spacing to the D access')
    def wire(layer,x,y,xx,yy,width=70.):
        b.rect(layer,min(x,xx)-width/2,min(y,yy)-width/2,max(x,xx)+width/2,max(y,yy)+width/2)
    def via(n,x,y):
        # Same cuts and landings as the verified TIME_CONTROL lift().
        if n>=7:b.sq(f'v{n}',x,y,400)
        else:b.via(n,x,y)
        for k in (n,n+1):
            if k==1:b.rect('m1',x-70,y-35,x+70,y+35)
            elif k==2:
                if n==2:b.rect('m2',x-35,y-70,x+35,y+70)
                else:b.rect('m2',x-70,y-35,x+70,y+35)
            elif k==3:b.rect('m3',x-35,y-70,x+35,y+70)
            else:b.sq(f'm{k}',x,y,400 if k==7 else 405 if k==8 else 145)
    top=(ROWS-1)*ROW_PITCH+h;right=COLUMNS*pitch-GAP
    terminals={};physical=[];placed=[];pins={}
    for path,c,mapping in registers:
        j,r=slots[path];x=j*pitch;y=(ROWS-1-r)*ROW_PITCH
        b.cell.insert(pya.CellInstArray(master.cell_index(),pya.Trans(u(x),u(y))))
        placed.append(dict(instance=path,cell=master.name,family='DFF',x_nm=x,y_nm=y,width_nm=w,height_nm=h,column=j,row=r))
        terminals.update({path+'/'+n:(x+px,y+py,l) for n,(px,py,l) in view.device_pins.items()})
        for f in view.physical_fingers:
            f=dict(f,device=path+'/'+f['device'])
            if 'gate_xy_nm' in f:f['gate_xy_nm']=(x+f['gate_xy_nm'][0],y+f['gate_xy_nm'][1])
            physical.append(f)
        if j<COLUMNS-1:
            # OpenRAM-style register abutment: continuous wells, M1 rails
            # and the shared CLK M6 rail across the access gap.
            b.rect('pwell',x+w,y+100,x+pitch,y+780);b.rect('nwell',x+w,y+780,x+pitch,y+1700)
            for phase in (200,1600):wire('m1',x+w-35,y+phase,x+pitch+35,y+phase)
            wire('m6',x+w-100,y+100,x+pitch+100,y+100,145.)
        # Supply pins join the row's M2 rails exactly as in TIME_CONTROL.
        for net,ty in (('VSS',y+90),('VDD',y+1600)):
            px,py,_=view.pins[net];px+=x;py+=y
            via(1,px,py);wire('m2',px,py,px,ty)
            if net=='VSS':b.rect('m2',px-70,ty-35,px+70,py+35)
        q=mapping['Q'];d=mapping['D']
        tx=x+tracks[r];qy=y+yq
        if r:wire('m7',tx,qy,x+xq,qy,400.)
        via(7,tx,qy);wire('m8',tx,qy,tx,top+top_margin,405.)
        pins['Q:'+q]=(tx,top+top_margin,'m8')
        via(7,x+xd,y+yd);pins['D:'+d]=(x+xd,y+yd,'m8')
    rails={}
    for r in range(ROWS):
        y=(ROWS-1-r)*ROW_PITCH
        for net,ty,start in (('VSS',y+90,-300.),('VDD',y+1600,-600.)):
            wire('m2',start,ty,right+300,ty);via(2,start,ty);rails[net,r]=(right+300,ty,'m2')
        # Row clock rails continue left to the shared M5 trunk.
        wire('m6',clock_x,y+100,100,y+100,145.);via(5,clock_x,y+100)
    wire('m5',clock_x,100,clock_x,(ROWS-1)*ROW_PITCH+100,145.)
    for net,x,ty in (('VSS',-300.,90.),('VDD',-600.,1600.)):
        wire('m3',x,ty,x,(ROWS-1)*ROW_PITCH+ty)
    pins.update(CLK=(clock_x,(ROWS-1)*ROW_PITCH+100,'m5'),VSS=(-300.,(ROWS-1)*ROW_PITCH+90,'m3'),
                VDD=(-600.,(ROWS-1)*ROW_PITCH+1600,'m3'))
    v=CellView(component,b,right,top,pins,terminals,[],[],[])
    v.physical_fingers=physical;v.placements=placed;v.row_rails=rails;v.q_tracks=tracks
    v.standard_family='DATA_REGISTERS'
    return v
