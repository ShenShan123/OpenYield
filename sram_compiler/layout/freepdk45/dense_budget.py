"""Area ceilings measured from the recorded OpenRAM reference GDS."""
import json
from .openram_core import ASSETS


def compare(views,rows,cols,bank):
    ref=json.loads((ASSETS/'area_reference.json').read_text())
    def area(name):
        key=name if name in ref['cells'] else 'openram_1rw_32x32_freepdk45_'+name
        return ref['cells'][key]['area_um2']
    caps={'precharge':(area('precharge_0'),'precharge_0'),
          'sense':(area('sense_amp'),'sense_amp'),
          'write':(area('write_driver'),'write_driver'),
          'buffer':(2*area('pinv'),'sum of two pinv physical bounding boxes'),
          'replica_driver':(area('wordline_driver'),'wordline_driver'),
          'latch':(ref['dff']['area_um2'],'dff (area ceiling; different storage topology)')}
    if (rows,cols)==(32,32) and bank is not None:
        caps.update(core=(area('capped_replica_bitcell_array'),'capped_replica_bitcell_array'),
            decoder=(area('hierarchical_decoder')+area('wordline_driver_array'),
                     'sum of hierarchical_decoder and wordline_driver_array'),
            bank=(ref['bank_area_um2'],'bank'))
    result=[]
    # Larger driver classes have no recorded reference; callers omit them.
    for name,(limit,reference) in caps.items():
        if name!='bank' and name not in views:continue
        v=bank if name=='bank' else views[name]
        measured=v.builder.cell.dbbox().area()
        result.append(dict(component=name,area_um2=measured,reference=reference,limit_um2=limit,
                           within_limit=measured<=limit+1e-9))
        if measured>limit+1e-9:raise ValueError(f'{name}: physical area {measured} exceeds OpenRAM ceiling {limit}')
    if (rows,cols)==(32,32) and 'decoder' in views:
        v=views['decoder'];box=v.builder.cell.dbbox();split=v.driver_start_nm/1000
        for name,measured,reference in [
                ('decoder_only',(split-box.left)*box.height(),'hierarchical_decoder'),
                ('wordline_drivers',(box.right-split)*box.height(),'wordline_driver_array')]:
            limit=area(reference)
            result.append(dict(component=name,area_um2=measured,reference=reference,limit_um2=limit,within_limit=measured<=limit))
            if measured>limit:raise ValueError(f'{name}: exceeds OpenRAM area ceiling')
    return dict(reference_gds_sha256=ref['reference_gds_sha256'],metric=ref['metric'],comparisons=result,
                matched_bank_reference=(rows,cols)==(32,32) and bank is not None)
