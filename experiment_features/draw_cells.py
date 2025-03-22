import schemdraw
import schemdraw.elements as elm

def draw_inverter():
    """Draw a CMOS inverter schematic using schemdraw"""
    d = schemdraw.Drawing(unit=2.5, fontsize=12, dpi=100)
    
    # Add power and ground
    d += elm.Line().left().length(2).label('VDD', loc='left')
    d += elm.Line().down().length(2).label('GND', loc='bottom')
    
    # Draw PMOS transistor
    d.push()  # Save current position
    d += elm.transistors.PMos(
        arrow='in',  # PMOS has inward arrow
        gate_anchor='center',
        bulk=False
    ).theta(90).label('PMOS', loc='right')
    
    # Draw NMOS transistor
    d.pop()  # Return to saved position
    d += elm.transistors.NMos(
        arrow='out',  # NMOS has outward arrow
        gate_anchor='center',
        bulk=False
    ).down().label('NMOS', loc='right')
    
    # Connect transistors
    d += elm.Line().right().length(1).label('Input', loc='bottom')
    d += elm.Line().up().length(1).label('Output', loc='right')
    
    # Add input/output labels
    d += elm.Dot().label('IN', loc='left')
    d += elm.Dot().label('OUT', loc='right')
    
    return d

# Generate and save schematic
inverter_schematic = draw_inverter()
inverter_schematic.save('inverter_schematic.svg', transparent=True)
inverter_schematic.save('inverter_schematic.png', dpi=300)