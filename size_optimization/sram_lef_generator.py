import os
import math

def generate_sram_lef(word_size, num_words, pin_width, pin_pitch, output_dir="."):
    """
    生成SRAM的LEF文件
    """
    
    addr_width = math.ceil(math.log2(num_words))
    
    
    width = 31.3547 + word_size * 2.9918 + num_words * 0.0568
    height = 49.9556 + word_size * 1.0801 + num_words * 0.0830

    base_name = f"sram_{word_size}_{num_words}"
    
    macro_name = f"{base_name}"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开文件
    filename = os.path.join(output_dir, f"{macro_name}.lef")
    fid = open(filename, 'w')
    
    # 生成LEF文件
    _write_lef_header(fid, macro_name, width, height)
    _write_pins(fid, word_size, addr_width, width, height, pin_width, pin_pitch)
    _write_power_pins(fid, width, height)
    _write_obstructions(fid, width, height, word_size, addr_width)
    _write_lef_footer(fid, macro_name)
    
    fid.close()
    
    return filename, macro_name

def _write_lef_header(fid, name, width, height):

    fid.write('VERSION 5.4 ;\n')
    fid.write('NAMESCASESENSITIVE ON ;\n')
    fid.write('BUSBITCHARS "[]" ;\n')
    fid.write('DIVIDERCHAR "/" ;\n')
    fid.write('UNITS\n')
    fid.write('  DATABASE MICRONS 2000 ;\n')
    fid.write('END UNITS\n')
    fid.write(f'MACRO {name}\n')
    fid.write('   CLASS BLOCK ;\n')
    fid.write(f'   SIZE {width:.4f} BY {height:.4f} ;\n')
    fid.write('   SYMMETRY X Y R90 ;\n')

def _write_pins(fid, word_size, addr_width, width, height, pin_width, pin_pitch):

    

    _write_control_pins(fid, height, pin_width, pin_pitch)
    

    _write_address_pins(fid, addr_width, width, height, pin_width, pin_pitch)
    

    _write_ioput_pins(fid, word_size, width, height, pin_width, pin_pitch)
    

def _write_control_pins(fid, height, pin_width, pin_pitch):
    """写入控制信号引脚"""
    y_start = height * 0.1
    control_pins = ['csb0', 'web0', 'clk0']
    
    for i, pin_name in enumerate(control_pins):
        y_pos = y_start + i * pin_pitch
        fid.write(f'   PIN {pin_name}\n')
        fid.write('      DIRECTION INPUT ;\n')
        fid.write('      PORT\n')
        fid.write('         LAYER metal3 ;\n')
        fid.write(f'         RECT  0.0 {y_pos:.4f} {pin_width:.4f} {y_pos+pin_width:.4f} ;\n')
        fid.write('      END\n')
        fid.write(f'   END {pin_name}\n')

def _write_address_pins(fid, addr_width, width, height, pin_width, pin_pitch):
    """写入地址引脚"""

    y_start = height * 0.6  # 从高度60%处开始
    
    for i in range(addr_width):
        y_pos = y_start + i * pin_pitch
        fid.write(f'   PIN addr0[{i}]\n')
        fid.write('      DIRECTION INPUT ;\n')
        fid.write('      PORT\n')
        fid.write('         LAYER metal3 ;\n')
        fid.write(f'         RECT  0.0 {y_pos:.4f} {pin_width:.4f} {y_pos+pin_width:.4f} ;\n')
        fid.write('      END\n')
        fid.write(f'   END addr0[{i}]\n')

def _write_ioput_pins(fid, word_size, width, height, pin_width, pin_pitch):

    x_start = width * 0.1

    right_out_count = int(word_size / 2)
    bottom_out_count = word_size - right_out_count

    required_width = (word_size + bottom_out_count) * pin_pitch

    y_start = height * 0.1

    bottom_available_width = width * 0.8
    if required_width > bottom_available_width:
    # 计算需要调整的引脚数量
        excess_width = required_width - bottom_available_width
        pins_to_adjust = math.ceil(excess_width / pin_pitch)
    else:
        pins_to_adjust = 0
    
    """写入数据输入引脚"""
    x_pos = x_start
    for i in range(word_size):
        x_pos = x_pos + pin_pitch
        if (i%2 == 1) and (pins_to_adjust > 0):
            x_pos = x_pos - pin_pitch + 2*pin_width
            pins_to_adjust = pins_to_adjust - 1
        fid.write(f'   PIN din0[{i}]\n')
        fid.write('      DIRECTION INPUT ;\n')
        fid.write('      PORT\n')
        fid.write('         LAYER metal4 ;\n')
        fid.write(f'         RECT  {x_pos:.4f} 0.0 {x_pos+pin_width:.4f} {pin_width:.4f} ;\n')
        fid.write('      END\n')
        fid.write(f'   END din0[{i}]\n')

    """写入数据输出引脚"""

    for i in range(bottom_out_count):
        x_pos = x_pos + pin_pitch
        if (i%2 == 1) and (pins_to_adjust > 0):
            x_pos = x_pos - pin_pitch + 2*pin_width
            pins_to_adjust = pins_to_adjust - 1
        fid.write(f'   PIN dout0[{right_out_count + i}]\n')
        fid.write('      DIRECTION OUTPUT ;\n')
        fid.write('      PORT\n')
        fid.write('         LAYER metal4 ;\n')
        fid.write(f'         RECT  {x_pos:.4f} 0.0 {x_pos+pin_width:.4f} {pin_width:.4f} ;\n')
        fid.write('      END\n')
        fid.write(f'   END dout0[{right_out_count + i}]\n')

    for i in range(right_out_count):
        y_pos = y_start + i * pin_pitch
        fid.write(f'   PIN dout0[{i}]\n')
        fid.write('      DIRECTION OUTPUT ;\n')
        fid.write('      PORT\n')
        fid.write('         LAYER metal3 ;\n')
        fid.write(f'         RECT  {width-pin_width:.4f} {y_pos:.4f} {width:.4f} {y_pos+pin_width:.4f} ;\n')
        fid.write('      END\n')
        fid.write(f'   END dout0[{i}]\n')

def _write_power_pins(fid, width, height):

    # VDD
    fid.write('   PIN vdd\n')
    fid.write('      DIRECTION INOUT ;\n')
    fid.write('      USE POWER ; \n')
    fid.write('      SHAPE ABUTMENT ; \n')
    fid.write('      PORT\n')
    fid.write('         LAYER metal4 ;\n')
    fid.write(f'         RECT  0.0 0.0 0.7 {height} ;\n')
    fid.write('         LAYER metal4 ;\n')
    fid.write(f'         RECT  {width-0.7:.4f} 0.0 {width} {height} ;\n')
    fid.write('         LAYER metal3 ;\n')
    fid.write(f'         RECT  0.0 0.0 {width} 0.7 ;\n')
    fid.write('         LAYER metal3 ;\n')
    fid.write(f'         RECT  0.0 {height-0.7:.4f} {width} {height} ;\n')
    fid.write('      END\n')
    fid.write('   END vdd\n')
    
    # GND
    fid.write('   PIN gnd\n')
    fid.write('      DIRECTION INOUT ;\n')
    fid.write('      USE GROUND ; \n')
    fid.write('      SHAPE ABUTMENT ; \n')
    fid.write('      PORT\n')
    fid.write('         LAYER metal3 ;\n')
    fid.write(f'         RECT  1.4 {height-2.1:.4f} {width-1.4:.4f} {height-1.4:.4f} ;\n')
    fid.write('         LAYER metal4 ;\n')
    fid.write(f'         RECT  {width-2.1:.4f} 1.4 {width-1.4:.4f} {height-1.4:.4f} ;\n')
    fid.write('         LAYER metal4 ;\n')
    fid.write(f'         RECT  1.4 1.4 2.1 {height-1.4:.4f} ;\n')
    fid.write('         LAYER metal3 ;\n')
    fid.write(f'         RECT  1.4 1.4 {width-1.4:.4f} 2.1 ;\n')
    fid.write('      END\n')
    fid.write('   END gnd\n')

def _write_obstructions(fid, width, height, word_size, addr_width):

    fid.write('   OBS\n')
    
    # metal1和metal2全层覆盖
    fid.write('   LAYER  metal1 ;\n')
    fid.write(f'      RECT  0.14 0.14 {width-0.14:.4f} {height-0.14:.4f} ;\n')
    fid.write('   LAYER  metal2 ;\n')
    fid.write(f'      RECT  0.14 0.14 {width-0.14:.4f} {height-0.14:.4f} ;\n')
    
    # metal3和metal4按照metal1和metal2的方式布置阻塞层
    fid.write('   LAYER  metal3 ;\n')
    fid.write(f'      RECT  0.14 0.14 {width-0.14:.4f} {height-0.14:.4f} ;\n')
    fid.write('   LAYER  metal4 ;\n')
    fid.write(f'      RECT  0.14 0.14 {width-0.14:.4f} {height-0.14:.4f} ;\n')
    
    fid.write('   END\n')

def _write_lef_footer(fid, name):
    """写入LEF文件尾"""
    fid.write(f'END    {name}\n')
    fid.write('END    LIBRARY\n')

def main():

    print("SRAM LEF文件器")
    print("=" * 45)
    
    try:
        word_size = 12
        num_words = 1024

        pin_width = 0.14
        pin_pitch = 2.86
        
        if word_size <= 0 or num_words <= 0:
            print("错误：字宽和字数必须为正整数")
            return
        
        filename, macro_name = generate_sram_lef(word_size, num_words, pin_width, pin_pitch)
        
        print(f"\n成功生成LEF文件: {filename}")
        print(f"模块名称: {macro_name}")
        print(f"配置: {word_size}位宽 x {num_words}字")
        print(f"地址线宽度: {math.ceil(math.log2(num_words))}位")
        print(f"引脚宽度:{pin_width}, 引脚间距: {pin_pitch}")
        
    except ValueError:
        print("错误：请输入有效的数字")
    except Exception as e:
        print(f"生成过程中出现错误: {e}")

if __name__ == "__main__":
    main()